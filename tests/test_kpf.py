"""Tests for Amazon Ion binary writer and KPF Guided View packager."""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path
from PIL import Image

import pytest

from manga2kindle.models import (
    ConversionConfig,
    DeviceProfile,
    PageAsset,
    PanelBox,
    ReadingDirection,
)
from manga2kindle.packaging.ion import (
    ION_BVM,
    SYM_ACTION,
    SYM_ACTIONS,
    SYM_ALT_STYLE,
    SYM_CLIP,
    SYM_FILL_COLOR,
    SYM_FILL_OPACITY,
    SYM_FIT_BOTH,
    SYM_LEAF,
    SYM_MAGNIFY,
    SYM_MAGNIFY_TARGET,
    SYM_NODE_TYPE,
    SYM_ORDINAL,
    SYM_PAN_ZOOM_VIEWER,
    SYM_RESOURCE_ID,
    SYM_SOURCE_ID,
    SYM_STRUCTURE,
    SYM_STYLE,
    SYM_STYLE_NAME,
    SYM_TARGET_ID,
    YJ_SYMBOLS_MAX_ID,
    IonAnnotation,
    IonBinaryWriter,
    IonEidRef,
    IonFloat64,
    IonList,
    IonSExp,
    IonStruct,
    IonSymbol,
)
from manga2kindle.packaging.kpf import KpfBuilder, strip_fingerprints


class TestIonBinaryWriter:
    """Unit tests for low-level Amazon Ion 1.0 binary serialization."""

    def test_bvm_header(self) -> None:
        """Encoding with include_bvm=True must prepend the 4-byte Ion BVM."""
        data = IonBinaryWriter.encode(IonSymbol(10), include_bvm=True)
        assert data.startswith(ION_BVM)
        assert len(data) == 4 + 2
        assert data[:4] == b"\xe0\x01\x00\xea"

    def test_encode_varuint(self) -> None:
        """VarUInt must set MSB=1 only on the final byte of 7-bit groups."""
        assert IonBinaryWriter.encode_varuint(0) == bytes([0x80])
        assert IonBinaryWriter.encode_varuint(1) == bytes([0x81])
        assert IonBinaryWriter.encode_varuint(127) == bytes([0xFF])
        # 128 = (1 << 7) | 0 -> 0x01, 0x80
        assert IonBinaryWriter.encode_varuint(128) == bytes([0x01, 0x80])
        # 598 = (4 << 7) | 86 -> 0x04, 0xD6
        assert IonBinaryWriter.encode_varuint(598) == bytes([0x04, 0xD6])

    def test_encode_int(self) -> None:
        """Ion ints must support 0 (type 2, len 0), positive and negative values."""
        # 0 is type 2, len 0
        assert IonBinaryWriter.encode(0, include_bvm=False) == b"\x20"
        # 42 is type 2, len 1 -> 0x21, 0x2A
        assert IonBinaryWriter.encode(42, include_bvm=False) == b"\x21\x2a"
        # -42 is type 3, len 1 -> 0x31, 0x2A
        assert IonBinaryWriter.encode(-42, include_bvm=False) == b"\x31\x2a"
        # Large integer 300 -> 0x012C -> len 2 -> 0x22, 0x01, 0x2C
        assert IonBinaryWriter.encode(300, include_bvm=False) == b"\x22\x01\x2c"

    def test_encode_float64(self) -> None:
        """Float64 must be serialized as IEEE 754 big-endian double (type 4, len 8)."""
        val = IonFloat64(1.5)
        encoded = IonBinaryWriter.encode(val, include_bvm=False)
        assert encoded[0] == 0x48
        assert len(encoded) == 9

    def test_encode_string(self) -> None:
        """String must be UTF-8 encoded with type 8."""
        encoded = IonBinaryWriter.encode("hello", include_bvm=False)
        assert encoded == b"\x85hello"

    def test_encode_symbol(self) -> None:
        """Symbol must be serialized as type 7 with big-endian SID."""
        # $0 is b"\x70"
        assert IonBinaryWriter.encode(IonSymbol(0), include_bvm=False) == b"\x70"
        # $42 is b"\x71\x2a"
        assert IonBinaryWriter.encode(IonSymbol(42), include_bvm=False) == b"\x71\x2a"
        with pytest.raises(ValueError, match="non-negative"):
            IonBinaryWriter.encode_symbol(-1)

    def test_encode_eid_ref(self) -> None:
        """EID reference must serialize as $598::'eid_string' annotation."""
        encoded = IonBinaryWriter.encode(IonEidRef("c0"), include_bvm=False)
        # Type 14 (annotation)
        assert (encoded[0] >> 4) == 0x0E
        # Must contain $598 VarUInt (0x04, 0xD6) and "c0"
        assert bytes([0x04, 0xD6]) in encoded
        assert b"c0" in encoded

    def test_encode_list_and_sexp(self) -> None:
        """List (type 11) and S-expression (type 12) encoding."""
        lst = IonList([1, 2])
        encoded_list = IonBinaryWriter.encode(lst, include_bvm=False)
        assert (encoded_list[0] >> 4) == 0x0B

        sexp = IonSExp([1, 2])
        encoded_sexp = IonBinaryWriter.encode(sexp, include_bvm=False)
        assert (encoded_sexp[0] >> 4) == 0x0C

    def test_encode_struct(self) -> None:
        """Struct (type 13) with field VarUInt and values."""
        st = IonStruct([(10, "val")])
        encoded = IonBinaryWriter.encode(st, include_bvm=False)
        assert (encoded[0] >> 4) == 0x0D

    def test_encode_annotation(self) -> None:
        """Annotation (type 14) wrapping an inner Ion value; empty list raises ValueError."""
        annot = IonAnnotation([259], IonStruct([]))
        encoded = IonBinaryWriter.encode(annot, include_bvm=False)
        assert (encoded[0] >> 4) == 0x0E
        with pytest.raises(ValueError, match="at least one"):
            IonBinaryWriter.encode_annotation([], IonStruct([]))

    def test_ion_symbol_table(self) -> None:
        """$ion_symbol_table must import YJ_symbols v10 with max_id=842."""
        assert YJ_SYMBOLS_MAX_ID == 851
        sym_tab = IonBinaryWriter.build_ion_symbol_table()
        assert sym_tab.startswith(ION_BVM)
        assert b"YJ_symbols" in sym_tab
        assert b"yj.authoring.source_file_name" in sym_tab


class TestComputePanelZoomLayout:
    """Unit tests for Kindle Create screen-fit magnification geometry."""

    def test_width_constrained_panel(self) -> None:
        """Wide panel scales to full page width and centers vertically."""
        panel = PanelBox(x=573, y=991, w=475, h=446, ordinal=1)
        layout = KpfBuilder.compute_panel_zoom_layout(panel, page_w=1048, page_h=1696)
        expected_scale = 1048.0 / 475.0
        assert layout.scale == pytest.approx(expected_scale)
        assert layout.vp_w == pytest.approx(1048.0)
        assert layout.vp_h == pytest.approx(446.0 * expected_scale)
        assert layout.vp_left == pytest.approx(0.0)
        assert layout.vp_top == pytest.approx((1696.0 - 446.0 * expected_scale) / 2.0)
        assert layout.img_w == pytest.approx(1048.0 * expected_scale)
        assert layout.img_h == pytest.approx(1696.0 * expected_scale)
        assert layout.img_left == pytest.approx(-573.0 * expected_scale)
        assert layout.img_top == pytest.approx(-991.0 * expected_scale)

    def test_height_constrained_panel(self) -> None:
        """Tall narrow panel scales to full page height and centers horizontally."""
        panel = PanelBox(x=100, y=0, w=400, h=1696, ordinal=1)
        layout = KpfBuilder.compute_panel_zoom_layout(panel, page_w=1048, page_h=1696)
        assert layout.scale == pytest.approx(1.0)
        assert layout.vp_w == pytest.approx(400.0)
        assert layout.vp_h == pytest.approx(1696.0)
        assert layout.vp_left == pytest.approx(324.0)
        assert layout.vp_top == pytest.approx(0.0)
        assert layout.img_left == pytest.approx(-100.0)
        assert layout.img_top == pytest.approx(0.0)

    def test_full_page_panel(self) -> None:
        """Full-page splash/cover panel uses scale 1.0 with zero offset."""
        panel = PanelBox(x=0, y=0, w=1048, h=1696, ordinal=1)
        layout = KpfBuilder.compute_panel_zoom_layout(panel, page_w=1048, page_h=1696)
        assert layout.scale == pytest.approx(1.0)
        assert (layout.vp_left, layout.vp_top, layout.vp_w, layout.vp_h) == pytest.approx(
            (0.0, 0.0, 1048.0, 1696.0)
        )
        assert (layout.img_left, layout.img_top, layout.img_w, layout.img_h) == pytest.approx(
            (0.0, 0.0, 1048.0, 1696.0)
        )


class TestKpfBuilderIntegration:
    """Integration tests for KpfBuilder assembling valid .kpf ZIP archives."""

    @pytest.fixture
    def sample_pages(self, tmp_path: Path) -> list[PageAsset]:
        """Create 2 synthetic image files and PageAsset models."""
        img1_path = tmp_path / "page_001.png"
        img2_path = tmp_path / "page_002.png"

        Image.new("RGB", (1264, 1680), (255, 255, 255)).save(img1_path, "PNG")
        Image.new("RGB", (1264, 1680), (240, 240, 240)).save(img2_path, "PNG")

        page1 = PageAsset(
            index=0,
            source_path=img1_path,
            processed_path=img1_path,
            width=1264,
            height=1680,
            is_cover=True,
            panels=[
                PanelBox(x=632, y=50, w=580, h=750, ordinal=1),
                PanelBox(x=50, y=50, w=550, h=750, ordinal=2),
            ],
        )
        page2 = PageAsset(
            index=1,
            source_path=img2_path,
            processed_path=img2_path,
            width=1264,
            height=1680,
            is_cover=False,
            panels=[
                PanelBox(x=632, y=100, w=580, h=700, ordinal=1),
                PanelBox(x=50, y=850, w=550, h=700, ordinal=2),
            ],
        )
        return [page1, page2]

    @pytest.fixture
    def config(self, tmp_path: Path) -> ConversionConfig:
        """ConversionConfig for KPF building."""
        return ConversionConfig(
            input_path=tmp_path / "input.cbz",
            output_path=tmp_path / "output.kfx",
            profile=DeviceProfile.from_name("KPW5"),
            direction=ReadingDirection.RTL,
        )

    def test_build_kpf_zip_structure_and_sqlite(
        self,
        tmp_path: Path,
        sample_pages: list[PageAsset],
        config: ConversionConfig,
    ) -> None:
        """KpfBuilder.build must emit valid ZIP with book.kdf, book.kcb, res/rsrc*."""
        builder = KpfBuilder()
        output_kpf = tmp_path / "test_book.kpf"

        result_path = builder.build(
            pages=sample_pages,
            title="Test Manga",
            config=config,
            output_kpf=output_kpf,
            author="Author Name",
            language="en",
        )

        assert result_path == output_kpf
        assert output_kpf.is_file()

        # 1. Verify ZIP contents
        with zipfile.ZipFile(output_kpf, "r") as zf:
            namelist = zf.namelist()
            assert "book.kdf" in namelist
            assert "book.kcb" in namelist
            # Raw images
            rsrc_entries = [name for name in namelist if name.startswith("res/")]
            assert len(rsrc_entries) == 2

            # 2. Verify book.kcb format
            kcb_raw = zf.read("book.kcb").decode("utf-8")
            kcb_json = json.loads(kcb_raw)
            assert "content_hash" in kcb_json
            assert len(kcb_json["content_hash"]) == 32  # valid MD5 hex
            assert "metadata" in kcb_json
            assert kcb_json["metadata"].get("format") == "yj"
            # Must NOT contain unknown keys that warn in KFX Output
            assert "tool_name" not in kcb_json["metadata"]
            assert "tool_version" not in kcb_json["metadata"]
            assert "book_reading_direction" not in kcb_json["metadata"]
            assert "book_virtual_panelmovement" not in kcb_json["metadata"]

            # 3. Read book.kdf, verify fingerprint header, strip, and check SQLite
            kdf_wrapped = zf.read("book.kdf")
            # Fingerprint signature at offset 1024
            assert kdf_wrapped[1024:1028] == b"\xfa\x50\x0a\x5f"

            raw_sqlite = strip_fingerprints(kdf_wrapped)
            assert raw_sqlite.startswith(b"SQLite format 3\x00")

        # 4. Open stripped SQLite and verify exact tables and DDL
        db_path = tmp_path / "verified_book.sqlite"
        db_path.write_bytes(raw_sqlite)

        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            # Check table existence
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cur.fetchall()}
            assert {
                "capabilities",
                "fragments",
                "fragment_properties",
                "gc_fragment_properties",
                "gc_reachable",
            }.issubset(tables)

            # Check capabilities
            cur.execute("SELECT key, version FROM capabilities")
            caps = dict(cur.fetchall())
            assert caps.get("db.schema") == 1

            # Check fragments
            cur.execute("SELECT id, payload_type, payload_value FROM fragments")
            frags = {row[0]: (row[1], row[2]) for row in cur.fetchall()}

            # Standard fragments
            assert "$ion_symbol_table" in frags
            assert "max_id" in frags
            assert "book_navigation" in frags
            assert "book_metadata" in frags
            assert "content_features" in frags
            assert "document_data" in frags
            assert "metadata" in frags
            assert "yj.section_pid_count_map" in frags

            # Verify book_metadata payload has title and author
            book_meta_bytes = frags["book_metadata"][1]
            assert b"kindle_capability_metadata" in book_meta_bytes
            assert b"yj_publisher_panels" in book_meta_bytes
            assert b"Test Manga" in book_meta_bytes
            assert b"Author Name" in book_meta_bytes

            # Check bcRawMedia rows: payload_type is 'path', payload_value is UTF-8 bytes
            path_frags = {fid: pval for fid, (ptype, pval) in frags.items() if ptype == "path"}
            assert len(path_frags) == 2
            for fid, pval in path_frags.items():
                assert isinstance(pval, bytes)
                assert pval.startswith(b"res/")
                assert pval == f"res/{fid}".encode("utf-8")

            # Check fragment_properties
            cur.execute("SELECT id, key, value FROM fragment_properties")
            props = cur.fetchall()
            prop_keys = {(row[0], row[1], row[2]) for row in props}
            assert ("book_metadata", "element_type", "book_metadata") in prop_keys
            assert ("document_data", "element_type", "document_data") in prop_keys

            # Verify external_resource $165 location stores rsrc* (NOT res/rsrc*)
            ext_frags = {
                fid: frags[fid][1]
                for fid, k, v in prop_keys
                if k == "element_type" and v == "external_resource"
            }
            assert len(ext_frags) == 2
            for ext_blob in ext_frags.values():
                assert b"rsrc" in ext_blob
                assert b"res/rsrc" not in ext_blob

            # Verify $157 backdrop style fragments exist cleanly
            style_frags = {
                fid: frags[fid][1]
                for fid, k, v in prop_keys
                if k == "element_type" and v == "style"
            }
            assert len(style_frags) == 2
            style_annot = IonBinaryWriter.encode_varuint(SYM_STYLE)
            fill_color_kv = IonBinaryWriter.encode_varuint(SYM_FILL_COLOR) + IonBinaryWriter.encode_int(0)
            fill_opacity_kv = (
                IonBinaryWriter.encode_varuint(SYM_FILL_OPACITY) + IonBinaryWriter.encode_float64(1.0)
            )
            for st_id, st_blob in style_frags.items():
                assert style_annot in st_blob
                assert (
                    IonBinaryWriter.encode_varuint(SYM_STYLE_NAME)
                    + IonBinaryWriter.encode_eid_ref(st_id)
                ) in st_blob
                assert fill_color_kv in st_blob
                assert fill_opacity_kv in st_blob

            # Verify storyline references ONLY the image container i_cont (panels nest under i_cont)
            child_links = {(row[0], row[2]) for row in props if row[1] == "child"}
            cur.execute("SELECT id, payload_value FROM fragments WHERE id GLOB 'p[0-9]*' ORDER BY id")
            panel_nodes = cur.fetchall()
            cur.execute("SELECT id, payload_value FROM fragments WHERE id GLOB 't[0-9]*' ORDER BY id")
            target_nodes = cur.fetchall()
            cur.execute("SELECT id, payload_value FROM fragments WHERE id GLOB 'b[0-9]*' ORDER BY id")
            backdrop_nodes = cur.fetchall()
            cur.execute("SELECT id, payload_value FROM fragments WHERE id GLOB 'v[0-9]*' ORDER BY id")
            viewport_nodes = cur.fetchall()
            cur.execute("SELECT id, payload_value FROM fragments WHERE id GLOB 'z[0-9]*' ORDER BY id")
            zoom_img_nodes = cur.fetchall()
            assert len(panel_nodes) == 4
            assert len(target_nodes) == 4
            assert len(backdrop_nodes) == 4
            assert len(viewport_nodes) == 4
            assert len(zoom_img_nodes) == 4

            # Storyline children must NOT directly include src/tgt panel nodes
            for fid, child_id in child_links:
                if fid.startswith("l"):
                    assert not child_id.startswith("p")
                    assert not child_id.startswith("t")

            actions_field = IonBinaryWriter.encode_varuint(SYM_ACTIONS)
            magnify_kv = (
                IonBinaryWriter.encode_varuint(SYM_ACTION)
                + IonBinaryWriter.encode_symbol(SYM_MAGNIFY)
            )
            pan_zoom_field = IonBinaryWriter.encode_varuint(SYM_PAN_ZOOM_VIEWER)
            fit_both_sym = IonBinaryWriter.encode_symbol(SYM_FIT_BOTH)
            struct_annot = IonBinaryWriter.encode_varuint(SYM_STRUCTURE)
            magnify_target_kv = (
                IonBinaryWriter.encode_varuint(SYM_NODE_TYPE)
                + IonBinaryWriter.encode_symbol(SYM_MAGNIFY_TARGET)
            )
            clip_true_kv = IonBinaryWriter.encode_varuint(SYM_CLIP) + b"\x11"
            leaf_kv = (
                IonBinaryWriter.encode_varuint(SYM_NODE_TYPE)
                + IonBinaryWriter.encode_symbol(SYM_LEAF)
            )

            target_map = dict(target_nodes)
            backdrop_map = dict(backdrop_nodes)
            viewport_map = dict(viewport_nodes)
            zoom_img_map = dict(zoom_img_nodes)
            expected_ordinals = [1, 2, 1, 2]
            for (src_eid, p_val), expected_ord in zip(panel_nodes, expected_ordinals):
                suffix = src_eid[1:]
                tgt_eid = f"t{suffix}"
                bg_eid = f"b{suffix}"
                vp_eid = f"v{suffix}"
                zoom_img_eid = f"z{suffix}"

                # Trigger node: actions + ordinal + pixel geometry, NO $684
                assert struct_annot in p_val
                assert pan_zoom_field not in p_val
                assert actions_field in p_val
                assert magnify_kv in p_val
                ord_kv = (
                    IonBinaryWriter.encode_varuint(SYM_ORDINAL)
                    + IonBinaryWriter.encode_int(expected_ord)
                )
                assert ord_kv in p_val
                src_ref_kv = (
                    IonBinaryWriter.encode_varuint(SYM_SOURCE_ID)
                    + IonBinaryWriter.encode_eid_ref(src_eid)
                )
                assert src_ref_kv in p_val
                tgt_ref_kv = (
                    IonBinaryWriter.encode_varuint(SYM_TARGET_ID)
                    + IonBinaryWriter.encode_eid_ref(tgt_eid)
                )
                assert tgt_ref_kv in p_val

                # Target popup ($439) -> [backdrop ($157/$429), clipping viewport ($476: True) -> zoomed image ($271/$175)]
                t_val = target_map[tgt_eid]
                assert struct_annot in t_val
                assert magnify_target_kv in t_val
                assert fit_both_sym in t_val
                assert actions_field not in t_val
                assert (tgt_eid, bg_eid) in child_links
                assert (tgt_eid, vp_eid) in child_links

                b_val = backdrop_map[bg_eid]
                assert IonBinaryWriter.encode_varuint(SYM_STYLE) in b_val
                assert IonBinaryWriter.encode_varuint(SYM_ALT_STYLE) in b_val

                v_val = viewport_map[vp_eid]
                assert clip_true_kv in v_val
                assert (vp_eid, zoom_img_eid) in child_links

                z_val = zoom_img_map[zoom_img_eid]
                assert leaf_kv in z_val
                assert IonBinaryWriter.encode_varuint(SYM_RESOURCE_ID) in z_val

            # Each 2-panel page has 3 base EIDs + 5 per panel = 13 EIDs
            cur.execute("SELECT payload_value FROM fragments WHERE id='yj.section_pid_count_map'")
            pid_map = cur.fetchone()[0]
            assert b"\x21\x0d" in pid_map
        finally:
            conn.close()

    def test_ltr_direction_emits_376(
        self,
        tmp_path: Path,
        sample_pages: list[PageAsset],
    ) -> None:
        """LTR reading direction should encode layout mode $376 in document_data."""
        config = ConversionConfig(
            input_path=tmp_path / "input.cbz",
            output_path=tmp_path / "output.kfx",
            profile=DeviceProfile.from_name("KPW5"),
            direction=ReadingDirection.LTR,
        )
        builder = KpfBuilder()
        out_kpf = tmp_path / "ltr_book.kpf"
        builder.build(sample_pages, "LTR Manga", config, out_kpf)

        with zipfile.ZipFile(out_kpf, "r") as zf:
            raw_sqlite = strip_fingerprints(zf.read("book.kdf"))

        db_path = tmp_path / "ltr_book.sqlite"
        db_path.write_bytes(raw_sqlite)

        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT payload_value FROM fragments WHERE id='document_data'")
            doc_data = cur.fetchone()[0]
            # Symbol 376 (0x0178) and binding 442 (0x01BA) in document_data
            assert b"\x72\x01\x78" in doc_data
            assert b"\x72\x01\xba" in doc_data
            # Guided View source panels emit magnify actions without deprecated $684 pan_zoom_viewer
            cur.execute("SELECT payload_value FROM fragments WHERE id LIKE 'p%'")
            gv_nodes = [row[0] for row in cur.fetchall()]
            pan_zoom_field = IonBinaryWriter.encode_varuint(SYM_PAN_ZOOM_VIEWER)
            magnify_kv = (
                IonBinaryWriter.encode_varuint(SYM_ACTION)
                + IonBinaryWriter.encode_symbol(SYM_MAGNIFY)
            )
            assert len(gv_nodes) == 4
            for node in gv_nodes:
                assert pan_zoom_field not in node
                assert magnify_kv in node
        finally:
            conn.close()

    def test_page_with_empty_panels_builds_cleanly(
        self,
        tmp_path: Path,
        config: ConversionConfig,
    ) -> None:
        """Edge case: page with empty panels list still builds a valid KPF cleanly."""
        img_path = tmp_path / "empty_page.png"
        Image.new("RGB", (1264, 1680), (255, 255, 255)).save(img_path, "PNG")

        page_no_panels = PageAsset(
            index=0,
            source_path=img_path,
            processed_path=img_path,
            width=1264,
            height=1680,
            panels=[],
        )

        builder = KpfBuilder()
        out_kpf = tmp_path / "no_panels.kpf"
        builder.build([page_no_panels], "No Panels Manga", config, out_kpf)

        assert out_kpf.is_file()
        with zipfile.ZipFile(out_kpf, "r") as zf:
            raw_sqlite = strip_fingerprints(zf.read("book.kdf"))

        db_path = tmp_path / "no_panels.sqlite"
        db_path.write_bytes(raw_sqlite)

        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT payload_value FROM fragments WHERE id='yj.section_pid_count_map'")
            pid_map = cur.fetchone()[0]
            # Section PID count for page with 0 panels should be 3
            assert b"\x21\x03" in pid_map
        finally:
            conn.close()
