"""Fixed-layout KPF package builder with Guided View panel magnification."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from manga2kindle.models import (
    ConversionConfig,
    PageAsset,
    PanelBox,
    ReadingDirection,
)
from manga2kindle.packaging.ion import (
    ION_BVM,
    SYM_ABSOLUTE,
    SYM_ACTION,
    SYM_ACTIONS,
    SYM_ALT_STYLE,
    SYM_AUXILIARY_DATA,
    SYM_BINDING,
    SYM_BLOCK,
    SYM_BOOK_METADATA,
    SYM_BUCKET_INDEX,
    SYM_CATEGORIES,
    SYM_CATEGORY_NAME,
    SYM_CHILDREN,
    SYM_CLIP,
    SYM_CONTAINER,
    SYM_CONTENT_FEATURES,
    SYM_COUNT,
    SYM_DEFAULT,
    SYM_DOCUMENT_DATA,
    SYM_EID,
    SYM_EIDHASH_EID_SECTION_MAP,
    SYM_EXTERNAL_RESOURCE,
    SYM_FEATURES_LIST,
    SYM_FILL_COLOR,
    SYM_FILL_OPACITY,
    SYM_FIT_BOTH,
    SYM_FIT_TYPE,
    SYM_FIXED,
    SYM_FIXED_LAYOUT,
    SYM_FIXED_LAYOUT_MODE_LTR,
    SYM_FIXED_LAYOUT_MODE_RTL,
    SYM_FORMAT,
    SYM_FORMAT_VERSION,
    SYM_HEIGHT,
    SYM_IMAGE_HEIGHT,
    SYM_IMAGE_WIDTH,
    SYM_JPG,
    SYM_KEY,
    SYM_LAYOUT,
    SYM_LAYOUT_TYPE,
    SYM_LEAF,
    SYM_LEFT,
    SYM_LEFT_TO_RIGHT_BINDING,
    SYM_LEFT_TO_RIGHT_PAGE,
    SYM_LOCAL_SOURCE_FILE_NAME,
    SYM_LOCATION,
    SYM_MAGNIFY,
    SYM_MAGNIFY_TARGET,
    SYM_MAJOR_VERSION,
    SYM_METADATA,
    SYM_MINOR_VERSION,
    SYM_NAMESPACE,
    SYM_NODE_TYPE,
    SYM_ORDINAL,
    SYM_PAGE_HEIGHT,
    SYM_PAGE_PROGRESSION_DIR,
    SYM_PAGE_TEMPLATE_TYPE,
    SYM_PAGE_WIDTH,
    SYM_PERCENTAGE,
    SYM_PNG,
    SYM_POSITION_MAP,
    SYM_POSITION_TYPE,
    SYM_PROPERTIES,
    SYM_READING_ORDER_NAME,
    SYM_READING_ORDERS,
    SYM_RESOURCE_ID,
    SYM_RESOURCE_LIST_REF,
    SYM_RIGHT_TO_LEFT_BINDING,
    SYM_RIGHT_TO_LEFT_PAGE,
    SYM_SECTION,
    SYM_SECTION_CONTENT,
    SYM_SECTION_ID,
    SYM_SECTION_PID_COUNT_MAP,
    SYM_SECTION_POSITION_ID_MAP,
    SYM_SECTIONS,
    SYM_SELF_REF,
    SYM_SOURCE_ID,
    SYM_STORYLINE,
    SYM_STORYLINE_ID,
    SYM_STRUCTURE,
    SYM_STYLE,
    SYM_STYLE_NAME,
    SYM_SYS_MAX_ID,
    SYM_SYS_VERSION,
    SYM_TARGET_ID,
    SYM_TOP,
    SYM_UNIT,
    SYM_VALUE,
    SYM_WIDTH,
    YJ_SYMBOLS_MAX_ID,
    IonAnnotation,
    IonBinaryWriter,
    IonEidRef,
    IonFloat64,
    IonList,
    IonStruct,
    IonSymbol,
)

_BASE32_CHARS = "0123456789ABCDEFGHJKMNPRSTUVWXYZ"
_BASE32_LEN = len(_BASE32_CHARS)

_FINGERPRINT_OFFSET = 1024
_FINGERPRINT_RECORD_LEN = 1024
_DATA_RECORD_LEN = 1024
_DATA_RECORD_COUNT = 1024
_FINGERPRINT_SIGNATURE = b"\xfa\x50\x0a\x5f"


def _int_to_base32(n: int) -> str:
    """Convert non-negative integer to Kindle base-32 string."""
    if n == 0:
        return _BASE32_CHARS[0]
    digits: list[str] = []
    while n > 0:
        digits.append(_BASE32_CHARS[n % _BASE32_LEN])
        n //= _BASE32_LEN
    return "".join(reversed(digits))


class IdAllocator:
    """Sequential ID generator for Kindle KDF fragments and nodes."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}

    def next_id(self, prefix: str) -> str:
        """Return the next sequential base-32 ID for prefix."""
        idx = self._counters.get(prefix, 0)
        self._counters[prefix] = idx + 1
        return prefix + _int_to_base32(idx)

    @property
    def total_count(self) -> int:
        """Total number of IDs allocated across all prefixes."""
        return sum(self._counters.values())


def add_fingerprints(data: bytes) -> bytes:
    """Insert 1024-byte fingerprint records at offset 1024 and every 1 MiB."""
    result = bytearray(data)
    offset = _FINGERPRINT_OFFSET
    while offset <= len(result):
        fp_record = bytearray(_FINGERPRINT_RECORD_LEN)
        fp_record[0:4] = _FINGERPRINT_SIGNATURE
        fp_record[4] = 0x01
        fp_record[5] = 0x00
        fp_record[6] = 0x00
        fp_record[7] = 0x40
        result[offset:offset] = fp_record
        offset += _FINGERPRINT_RECORD_LEN + _DATA_RECORD_LEN * _DATA_RECORD_COUNT
    return bytes(result)


def strip_fingerprints(data: bytes) -> bytes:
    """Remove fingerprint records from SQLite database bytes."""
    result = bytearray(data)
    offset = _FINGERPRINT_OFFSET
    while offset < len(result):
        if result[offset : offset + 4] == _FINGERPRINT_SIGNATURE:
            del result[offset : offset + _FINGERPRINT_RECORD_LEN]
            offset += _DATA_RECORD_LEN * _DATA_RECORD_COUNT
        else:
            break
    return bytes(result)


def _dim_pct(val: float | int, total: float | int) -> IonStruct:
    """Encode percentage dimension struct {$307: float, $306: $314}."""
    pct = round(100.0 * float(val) / float(total), 4) if total > 0 else 0.0
    return IonStruct([
        (SYM_VALUE, IonFloat64(pct)),
        (SYM_UNIT, IonSymbol(SYM_PERCENTAGE)),
    ])


@dataclass(frozen=True)
class PanelZoomLayout:
    """Screen-fit viewport and zoomed image placement for a Guided View panel."""

    scale: float
    vp_left: float
    vp_top: float
    vp_w: float
    vp_h: float
    img_left: float
    img_top: float
    img_w: float
    img_h: float


class KpfBuilder:
    """Fixed-layout KPF package builder with Guided View panel magnification."""

    @staticmethod
    def compute_panel_zoom_layout(
        panel: PanelBox,
        page_w: int,
        page_h: int,
    ) -> PanelZoomLayout:
        """Compute Kindle Create screen-fit clipping viewport and zoomed image coordinates."""
        scale = min(float(page_w) / float(panel.w), float(page_h) / float(panel.h))
        vp_w = float(panel.w) * scale
        vp_h = float(panel.h) * scale
        vp_left = (float(page_w) - vp_w) / 2.0
        vp_top = (float(page_h) - vp_h) / 2.0
        return PanelZoomLayout(
            scale=scale,
            vp_left=vp_left,
            vp_top=vp_top,
            vp_w=vp_w,
            vp_h=vp_h,
            img_left=-float(panel.x) * scale,
            img_top=-float(panel.y) * scale,
            img_w=float(page_w) * scale,
            img_h=float(page_h) * scale,
        )

    def build(
        self,
        pages: list[PageAsset],
        title: str,
        config: ConversionConfig,
        output_kpf: Path,
        author: str | None = None,
        language: str = "en",
    ) -> Path:
        """Assemble processed manga pages and panel boxes into a fixed-layout KPF."""
        if not pages:
            raise ValueError("Cannot build KPF without at least one page")

        out_path = Path(output_kpf)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ids = IdAllocator()
        res_list_id = ids.next_id("d")
        # Single-letter prefix 'a' is required by KFX Output SHORT symbol validation.
        bg_style_ids = (ids.next_id("a"), ids.next_id("a"))
        page_data = self._allocate_page_data(pages, ids)

        fragments: list[tuple[str, str, bytes]] = []
        frag_props: list[tuple[str, str, str]] = []
        gc_frag_props: list[tuple[str, str, str]] = []
        gc_reachable: set[str] = set()

        self._append_global_fragments(
            fragments=fragments,
            frag_props=frag_props,
            gc_frag_props=gc_frag_props,
            gc_reachable=gc_reachable,
            page_data=page_data,
            ids=ids,
            res_list_id=res_list_id,
            bg_style_ids=bg_style_ids,
            title=title,
            config=config,
            author=author,
            language=language,
        )

        eid_section_map: list[tuple[str, str]] = []
        section_pid_counts: list[tuple[str, int]] = []
        for pd in page_data:
            self._append_page_fragments(
                pd=pd,
                bg_style_ids=bg_style_ids,
                fragments=fragments,
                frag_props=frag_props,
                gc_frag_props=gc_frag_props,
                gc_reachable=gc_reachable,
                eid_section_map=eid_section_map,
                section_pid_counts=section_pid_counts,
            )

        self._append_eid_maps(
            fragments=fragments,
            frag_props=frag_props,
            gc_reachable=gc_reachable,
            eid_section_map=eid_section_map,
            section_pid_counts=section_pid_counts,
        )

        self._write_kdf_and_kpf(
            out_path=out_path,
            page_data=page_data,
            fragments=fragments,
            frag_props=frag_props,
            gc_frag_props=gc_frag_props,
            gc_reachable=gc_reachable,
        )
        return out_path

    @staticmethod
    def _record_blob(
        fragments: list[tuple[str, str, bytes]],
        frag_props: list[tuple[str, str, str]],
        gc_reachable: set[str],
        fid: str,
        element_type: str,
        ion_val: object,
        *,
        reachable: bool = True,
    ) -> None:
        """Encode an Ion value with BVM and register its fragment metadata."""
        fragments.append((fid, "blob", IonBinaryWriter.encode(ion_val, include_bvm=True)))
        frag_props.append((fid, "element_type", element_type))
        if reachable:
            gc_reachable.add(fid)

    @staticmethod
    def _allocate_page_data(pages: list[PageAsset], ids: IdAllocator) -> list[dict]:
        """Inspect page image dimensions/formats and allocate sequential KDF IDs."""
        page_data: list[dict] = []
        for page in pages:
            img_path = page.processed_path or page.source_path
            w, h = page.width, page.height
            if w <= 0 or h <= 0:
                with Image.open(img_path) as im:
                    w, h = im.size

            with open(img_path, "rb") as f:
                header = f.read(8)
            fmt_sym = SYM_PNG if header.startswith(b"\x89PNG\r\n\x1a\n") else SYM_JPG

            panels_info: list[dict] = [
                {
                    "panel": p,
                    "ordinal": p.ordinal if p.ordinal > 0 else (idx + 1),
                    "src_eid": ids.next_id("p"),
                    "tgt_eid": ids.next_id("t"),
                    "bg_eid": ids.next_id("b"),
                    "vp_eid": ids.next_id("v"),
                    "zoom_img_eid": ids.next_id("z"),
                }
                for idx, p in enumerate(page.panels)
            ]

            page_data.append({
                "page": page,
                "img_path": img_path,
                "width": w,
                "height": h,
                "fmt_sym": fmt_sym,
                "sid": ids.next_id("s"),
                "t_eid": ids.next_id("c"),
                "l_id": ids.next_id("l"),
                "i_cont": ids.next_id("i"),
                "i_leaf": ids.next_id("i"),
                "e_id": ids.next_id("e"),
                "rsrc_id": ids.next_id("rsrc"),
                "d_id": ids.next_id("d"),
                "panels": panels_info,
            })
        return page_data

    def _append_global_fragments(
        self,
        *,
        fragments: list[tuple[str, str, bytes]],
        frag_props: list[tuple[str, str, str]],
        gc_frag_props: list[tuple[str, str, str]],
        gc_reachable: set[str],
        page_data: list[dict],
        ids: IdAllocator,
        res_list_id: str,
        bg_style_ids: tuple[str, str],
        title: str,
        config: ConversionConfig,
        author: str | None,
        language: str,
    ) -> None:
        """Append symbol table, book metadata, content features, and document headers."""
        cover_page = next((pd for pd in page_data if pd["page"].is_cover), page_data[0])
        cover_resource_eid = cover_page["e_id"]

        fragments.append(("$ion_symbol_table", "blob", IonBinaryWriter.build_ion_symbol_table()))
        frag_props.append(("$ion_symbol_table", "element_type", "$ion_symbol_table"))
        gc_reachable.add("$ion_symbol_table")

        self._record_blob(
            fragments, frag_props, gc_reachable,
            "max_id", "max_id", YJ_SYMBOLS_MAX_ID + 3,
        )

        fragments.append(("book_navigation", "blob", ION_BVM))
        frag_props.append(("book_navigation", "element_type", "book_navigation"))
        gc_frag_props.append(("book_navigation", "child", "book_navigation"))
        gc_reachable.add("book_navigation")

        # book_metadata
        book_id = "P_" + uuid.uuid4().hex[:21]
        cap_category = IonStruct([
            (SYM_CATEGORY_NAME, "kindle_capability_metadata"),
            (SYM_METADATA, IonList([
                IonStruct([(SYM_KEY, "yj_publisher_panels"), (SYM_VALUE, 1)]),
                IonStruct([(SYM_KEY, "yj_fixed_layout"), (SYM_VALUE, 1)]),
            ])),
        ])
        title_kvs: list[IonStruct] = [
            IonStruct([(SYM_KEY, "book_id"), (SYM_VALUE, book_id)]),
            IonStruct([(SYM_KEY, "language"), (SYM_VALUE, language)]),
            IonStruct([(SYM_KEY, "title"), (SYM_VALUE, title)]),
            IonStruct([(SYM_KEY, "cover_image"), (SYM_VALUE, cover_resource_eid)]),
        ]
        if author:
            title_kvs.append(IonStruct([(SYM_KEY, "author"), (SYM_VALUE, author)]))

        title_category = IonStruct([
            (SYM_CATEGORY_NAME, "kindle_title_metadata"),
            (SYM_METADATA, IonList(title_kvs)),
        ])
        ebook_category = IonStruct([
            (SYM_CATEGORY_NAME, "kindle_ebook_metadata"),
            (SYM_METADATA, IonList([IonStruct([(SYM_KEY, "selection"), (SYM_VALUE, "enabled")])])),
        ])
        audit_category = IonStruct([
            (SYM_CATEGORY_NAME, "kindle_audit_metadata"),
            (SYM_METADATA, IonList([
                IonStruct([(SYM_KEY, "file_creator"), (SYM_VALUE, "KC")]),
                IonStruct([(SYM_KEY, "creator_version"), (SYM_VALUE, "1.110.0.0")]),
            ])),
        ])
        bm_ion = IonAnnotation([SYM_BOOK_METADATA], IonStruct([
            (SYM_CATEGORIES, IonList([cap_category, title_category, ebook_category, audit_category])),
        ]))
        self._record_blob(fragments, frag_props, gc_reachable, "book_metadata", "book_metadata", bm_ion)

        # content_features
        features = [
            IonStruct([
                (SYM_NAMESPACE, "com.amazon.yjconversion"),
                (SYM_KEY, feat_key),
                (SYM_PROPERTIES, IonStruct([
                    (SYM_SYS_VERSION, IonStruct([(SYM_MAJOR_VERSION, 2), (SYM_MINOR_VERSION, 0)])),
                ])),
            ])
            for feat_key in ("yj_non_pdf_fixed_layout", "yj_publisher_panels")
        ]
        cf_ion = IonAnnotation([SYM_CONTENT_FEATURES], IonStruct([
            (SYM_SELF_REF, IonSymbol(SYM_CONTENT_FEATURES)),
            (SYM_FEATURES_LIST, IonList(features)),
        ]))
        self._record_blob(
            fragments, frag_props, gc_reachable,
            "content_features", "content_features", cf_ion,
        )

        # document_data & metadata
        is_rtl = config.direction == ReadingDirection.RTL
        layout_mode_sym = SYM_FIXED_LAYOUT_MODE_RTL if is_rtl else SYM_FIXED_LAYOUT_MODE_LTR
        page_dir_sym = SYM_RIGHT_TO_LEFT_PAGE if is_rtl else SYM_LEFT_TO_RIGHT_PAGE
        binding_sym = SYM_RIGHT_TO_LEFT_BINDING if is_rtl else SYM_LEFT_TO_RIGHT_BINDING

        reading_order = IonStruct([
            (SYM_READING_ORDER_NAME, IonSymbol(SYM_DEFAULT)),
            (SYM_SECTIONS, IonList([IonEidRef(pd["sid"]) for pd in page_data])),
        ])
        doc_data_ion = IonAnnotation([SYM_DOCUMENT_DATA], IonStruct([
            (SYM_FORMAT_VERSION, IonFloat64(16.0)),
            (SYM_PAGE_PROGRESSION_DIR, IonSymbol(page_dir_sym)),
            (SYM_SYS_MAX_ID, ids.total_count),
            (SYM_LAYOUT, IonSymbol(layout_mode_sym)),
            (SYM_BINDING, IonSymbol(binding_sym)),
            (SYM_AUXILIARY_DATA, IonStruct([(SYM_RESOURCE_LIST_REF, IonEidRef(res_list_id))])),
            (SYM_READING_ORDERS, IonList([reading_order])),
        ]))
        self._record_blob(fragments, frag_props, gc_reachable, "document_data", "document_data", doc_data_ion)
        frag_props.append(("document_data", "child", res_list_id))
        gc_frag_props.append(("document_data", "child", res_list_id))

        metadata_ion = IonAnnotation([SYM_METADATA], IonStruct([
            (SYM_READING_ORDERS, IonList([reading_order])),
        ]))
        self._record_blob(fragments, frag_props, gc_reachable, "metadata", "metadata", metadata_ion)

        res_list_ion = IonAnnotation([SYM_AUXILIARY_DATA], IonStruct([
            (SYM_SELF_REF, IonEidRef(res_list_id)),
            (SYM_METADATA, IonList([
                IonStruct([
                    (SYM_KEY, "auxData_resource_list"),
                    (SYM_VALUE, IonList([IonEidRef(pd["d_id"]) for pd in page_data])),
                ]),
            ])),
        ]))
        self._record_blob(fragments, frag_props, gc_reachable, res_list_id, "auxiliary_data", res_list_ion)

        # Backdrop style fragments ($157) shared by Guided View popup containers
        for st_id in bg_style_ids:
            st_ion = IonAnnotation([SYM_STYLE], IonStruct([
                (SYM_STYLE_NAME, IonEidRef(st_id)),
                (SYM_FILL_COLOR, 0),
                (SYM_FILL_OPACITY, IonFloat64(1.0)),
            ]))
            self._record_blob(fragments, frag_props, gc_reachable, st_id, "style", st_ion)

    def _append_panel_fragments(
        self,
        *,
        p_info: dict,
        page_w: int,
        page_h: int,
        e_id: str,
        bg_style_ids: tuple[str, str],
        fragments: list[tuple[str, str, bytes]],
        frag_props: list[tuple[str, str, str]],
        gc_reachable: set[str],
    ) -> list[str]:
        """Append Guided View trigger, $439 popup target, backdrop, viewport, and zoom image."""
        p_box: PanelBox = p_info["panel"]
        ordinal: int = p_info["ordinal"]
        src_eid: str = p_info["src_eid"]
        tgt_eid: str = p_info["tgt_eid"]
        bg_eid: str = p_info["bg_eid"]
        vp_eid: str = p_info["vp_eid"]
        zoom_img_eid: str = p_info["zoom_img_eid"]
        bg_style_id, bg_alt_style_id = bg_style_ids

        layout = self.compute_panel_zoom_layout(p_box, page_w, page_h)
        pct_100 = _dim_pct(100, 100)
        pct_0 = _dim_pct(0, 100)

        activate_act = IonStruct([
            (SYM_ACTION, IonSymbol(SYM_MAGNIFY)),
            (SYM_SOURCE_ID, IonEidRef(src_eid)),
            (SYM_TARGET_ID, IonEidRef(tgt_eid)),
        ])
        src_node = IonAnnotation([SYM_STRUCTURE], IonStruct([
            (SYM_SELF_REF, IonEidRef(src_eid)),
            (SYM_POSITION_TYPE, IonSymbol(SYM_ABSOLUTE)),
            (SYM_FIT_TYPE, IonSymbol(SYM_FIT_BOTH)),
            (SYM_WIDTH, IonFloat64(float(p_box.w))),
            (SYM_HEIGHT, IonFloat64(float(p_box.h))),
            (SYM_TOP, IonFloat64(float(p_box.y))),
            (SYM_LEFT, IonFloat64(float(p_box.x))),
            (SYM_ACTIONS, IonList([activate_act])),
            (SYM_ORDINAL, ordinal),
            (SYM_NODE_TYPE, IonSymbol(SYM_CONTAINER)),
        ]))
        self._record_blob(fragments, frag_props, gc_reachable, src_eid, "structure", src_node)

        tgt_node = IonAnnotation([SYM_STRUCTURE], IonStruct([
            (SYM_SELF_REF, IonEidRef(tgt_eid)),
            (SYM_WIDTH, pct_100),
            (SYM_HEIGHT, pct_100),
            (SYM_POSITION_TYPE, IonSymbol(SYM_ABSOLUTE)),
            (SYM_NODE_TYPE, IonSymbol(SYM_MAGNIFY_TARGET)),
            (SYM_FIT_TYPE, IonSymbol(SYM_FIT_BOTH)),
            (SYM_CHILDREN, IonList([IonEidRef(bg_eid), IonEidRef(vp_eid)])),
        ]))
        self._record_blob(fragments, frag_props, gc_reachable, tgt_eid, "structure", tgt_node)
        frag_props.append((tgt_eid, "child", bg_eid))
        frag_props.append((tgt_eid, "child", vp_eid))

        bg_node = IonAnnotation([SYM_STRUCTURE], IonStruct([
            (SYM_SELF_REF, IonEidRef(bg_eid)),
            (SYM_WIDTH, pct_100),
            (SYM_HEIGHT, pct_100),
            (SYM_TOP, pct_0),
            (SYM_LEFT, pct_0),
            (SYM_STYLE, IonEidRef(bg_style_id)),
            (SYM_ALT_STYLE, IonEidRef(bg_alt_style_id)),
            (SYM_NODE_TYPE, IonSymbol(SYM_CONTAINER)),
        ]))
        self._record_blob(fragments, frag_props, gc_reachable, bg_eid, "structure", bg_node)
        frag_props.append((bg_eid, "child", bg_style_id))
        frag_props.append((bg_eid, "child", bg_alt_style_id))

        vp_node = IonAnnotation([SYM_STRUCTURE], IonStruct([
            (SYM_SELF_REF, IonEidRef(vp_eid)),
            (SYM_POSITION_TYPE, IonSymbol(SYM_ABSOLUTE)),
            (SYM_FIT_TYPE, IonSymbol(SYM_FIT_BOTH)),
            (SYM_WIDTH, IonFloat64(layout.vp_w)),
            (SYM_HEIGHT, IonFloat64(layout.vp_h)),
            (SYM_TOP, IonFloat64(layout.vp_top)),
            (SYM_LEFT, IonFloat64(layout.vp_left)),
            (SYM_CLIP, True),
            (SYM_NODE_TYPE, IonSymbol(SYM_CONTAINER)),
            (SYM_CHILDREN, IonList([IonEidRef(zoom_img_eid)])),
        ]))
        self._record_blob(fragments, frag_props, gc_reachable, vp_eid, "structure", vp_node)
        frag_props.append((vp_eid, "child", zoom_img_eid))

        zoom_img_node = IonAnnotation([SYM_STRUCTURE], IonStruct([
            (SYM_SELF_REF, IonEidRef(zoom_img_eid)),
            (SYM_POSITION_TYPE, IonSymbol(SYM_ABSOLUTE)),
            (SYM_FIT_TYPE, IonSymbol(SYM_FIT_BOTH)),
            (SYM_WIDTH, IonFloat64(layout.img_w)),
            (SYM_HEIGHT, IonFloat64(layout.img_h)),
            (SYM_TOP, IonFloat64(layout.img_top)),
            (SYM_LEFT, IonFloat64(layout.img_left)),
            (SYM_NODE_TYPE, IonSymbol(SYM_LEAF)),
            (SYM_RESOURCE_ID, IonEidRef(e_id)),
        ]))
        self._record_blob(fragments, frag_props, gc_reachable, zoom_img_eid, "structure", zoom_img_node)
        frag_props.append((zoom_img_eid, "child", e_id))

        return [src_eid, tgt_eid, bg_eid, vp_eid, zoom_img_eid]

    def _append_page_fragments(
        self,
        *,
        pd: dict,
        bg_style_ids: tuple[str, str],
        fragments: list[tuple[str, str, bytes]],
        frag_props: list[tuple[str, str, str]],
        gc_frag_props: list[tuple[str, str, str]],
        gc_reachable: set[str],
        eid_section_map: list[tuple[str, str]],
        section_pid_counts: list[tuple[str, int]],
    ) -> None:
        """Append section, storyline, image structure, Guided View panels, and media rows for one page."""
        sid = pd["sid"]
        t_eid = pd["t_eid"]
        l_id = pd["l_id"]
        w = pd["width"]
        h = pd["height"]
        i_cont = pd["i_cont"]
        i_leaf = pd["i_leaf"]
        e_id = pd["e_id"]
        rsrc_id = pd["rsrc_id"]
        d_id = pd["d_id"]
        panels = pd["panels"]

        # Section
        inline_struct = IonAnnotation([SYM_STRUCTURE], IonStruct([
            (SYM_SELF_REF, IonEidRef(t_eid)),
            (SYM_STORYLINE_ID, IonEidRef(l_id)),
            (SYM_PAGE_WIDTH, w),
            (SYM_PAGE_HEIGHT, h),
            (SYM_LAYOUT_TYPE, IonSymbol(SYM_FIXED_LAYOUT)),
            (SYM_PAGE_TEMPLATE_TYPE, IonSymbol(SYM_FIXED)),
            (SYM_NODE_TYPE, IonSymbol(SYM_CONTAINER)),
        ]))
        section_ion = IonAnnotation([SYM_SECTION], IonStruct([
            (SYM_SECTION_ID, IonEidRef(sid)),
            (SYM_SECTION_CONTENT, IonList([inline_struct])),
        ]))
        self._record_blob(fragments, frag_props, gc_reachable, sid, "section", section_ion)
        frag_props.append((sid, "child", l_id))

        # Storyline contains ONLY the page image container; panels nest inside i_cont
        story_ion = IonAnnotation([SYM_STORYLINE], IonStruct([
            (SYM_STORYLINE_ID, IonEidRef(l_id)),
            (SYM_CHILDREN, IonList([IonEidRef(i_cont)])),
        ]))
        self._record_blob(fragments, frag_props, gc_reachable, l_id, "storyline", story_ion)
        for cid in (i_cont, l_id):
            frag_props.append((l_id, "child", cid))
            gc_frag_props.append((l_id, "child", cid))

        # Page container holds leaf image followed by interleaved src/tgt panel triggers & popups
        cont_children = [i_leaf]
        for p_info in panels:
            cont_children.append(p_info["src_eid"])
            cont_children.append(p_info["tgt_eid"])

        cont_ion = IonAnnotation([SYM_STRUCTURE], IonStruct([
            (SYM_SELF_REF, IonEidRef(i_cont)),
            (SYM_WIDTH, w),
            (SYM_HEIGHT, h),
            (SYM_POSITION_TYPE, IonSymbol(SYM_ABSOLUTE)),
            (SYM_LAYOUT_TYPE, IonSymbol(SYM_BLOCK)),
            (SYM_NODE_TYPE, IonSymbol(SYM_CONTAINER)),
            (SYM_CHILDREN, IonList([IonEidRef(cid) for cid in cont_children])),
        ]))
        self._record_blob(fragments, frag_props, gc_reachable, i_cont, "structure", cont_ion)
        for cid in cont_children:
            frag_props.append((i_cont, "child", cid))

        leaf_ion = IonAnnotation([SYM_STRUCTURE], IonStruct([
            (SYM_SELF_REF, IonEidRef(i_leaf)),
            (SYM_WIDTH, w),
            (SYM_HEIGHT, h),
            (SYM_RESOURCE_ID, IonEidRef(e_id)),
            (SYM_POSITION_TYPE, IonSymbol(SYM_ABSOLUTE)),
            (SYM_NODE_TYPE, IonSymbol(SYM_LEAF)),
            (SYM_FIT_TYPE, IonSymbol(SYM_FIT_BOTH)),
        ]))
        self._record_blob(fragments, frag_props, gc_reachable, i_leaf, "structure", leaf_ion)
        frag_props.append((i_leaf, "child", e_id))

        panel_eids: list[str] = []
        for p_info in panels:
            panel_eids.extend(
                self._append_panel_fragments(
                    p_info=p_info,
                    page_w=w,
                    page_h=h,
                    e_id=e_id,
                    bg_style_ids=bg_style_ids,
                    fragments=fragments,
                    frag_props=frag_props,
                    gc_reachable=gc_reachable,
                )
            )

        # External resource & bcRawMedia row
        rsrc_rel_path = f"res/{rsrc_id}"
        ext_res_ion = IonAnnotation([SYM_EXTERNAL_RESOURCE], IonStruct([
            (SYM_LOCAL_SOURCE_FILE_NAME, pd["img_path"].name),
            (SYM_FORMAT, IonSymbol(pd["fmt_sym"])),
            (SYM_LOCATION, rsrc_id),
            (SYM_AUXILIARY_DATA, IonEidRef(d_id)),
            (SYM_IMAGE_WIDTH, IonFloat64(float(w))),
            (SYM_RESOURCE_ID, IonEidRef(e_id)),
            (SYM_IMAGE_HEIGHT, IonFloat64(float(h))),
        ]))
        self._record_blob(fragments, frag_props, gc_reachable, e_id, "external_resource", ext_res_ion)
        frag_props.append((e_id, "child", d_id))
        frag_props.append((e_id, "child", rsrc_id))

        fragments.append((rsrc_id, "path", rsrc_rel_path.encode("utf-8")))
        frag_props.append((rsrc_id, "element_type", "bcRawMedia"))
        gc_reachable.add(rsrc_id)

        img_stat = pd["img_path"].stat()
        aux_ion = IonAnnotation([SYM_AUXILIARY_DATA], IonStruct([
            (SYM_SELF_REF, IonEidRef(d_id)),
            (SYM_METADATA, IonList([
                IonStruct([(SYM_KEY, "type"), (SYM_VALUE, "resource")]),
                IonStruct([(SYM_KEY, "resource_stream"), (SYM_VALUE, rsrc_rel_path)]),
                IonStruct([(SYM_KEY, "size"), (SYM_VALUE, str(img_stat.st_size))]),
                IonStruct([(SYM_KEY, "modified_time"), (SYM_VALUE, str(int(img_stat.st_mtime)))]),
                IonStruct([(SYM_KEY, "location"), (SYM_VALUE, str(pd["img_path"]))]),
            ])),
        ]))
        self._record_blob(fragments, frag_props, gc_reachable, d_id, "auxiliary_data", aux_ion)

        spim_eids = [t_eid, i_cont, i_leaf] + panel_eids
        spm_ion = IonAnnotation([SYM_SECTION_POSITION_ID_MAP], IonStruct([
            (SYM_SECTION_ID, IonEidRef(sid)),
            (SYM_POSITION_MAP, IonList([
                IonList([idx + 1, IonEidRef(e)]) for idx, e in enumerate(spim_eids)
            ])),
        ]))
        self._record_blob(
            fragments, frag_props, gc_reachable,
            f"{sid}-spm", "section_position_id_map", spm_ion,
        )

        eid_section_map.extend([(sid, sid), (t_eid, sid), (i_cont, sid), (i_leaf, sid)])
        for panel_eid in panel_eids:
            eid_section_map.append((panel_eid, sid))
        section_pid_counts.append((sid, len(spim_eids)))

    def _append_eid_maps(
        self,
        *,
        fragments: list[tuple[str, str, bytes]],
        frag_props: list[tuple[str, str, str]],
        gc_reachable: set[str],
        eid_section_map: list[tuple[str, str]],
        section_pid_counts: list[tuple[str, int]],
    ) -> None:
        """Append EID-to-section hash buckets and section PID count summary."""
        num_buckets = 1 if len(eid_section_map) <= 10 else max(1, round(len(eid_section_map) / 10.3))
        buckets: dict[int, list[tuple[str, str]]] = {i: [] for i in range(num_buckets)}
        for eid_str, sec_id in eid_section_map:
            buckets[sum(ord(c) for c in eid_str) % num_buckets].append((eid_str, sec_id))

        for b_idx in range(num_buckets):
            bucket_ion = IonAnnotation([SYM_EIDHASH_EID_SECTION_MAP], IonStruct([
                (SYM_BUCKET_INDEX, b_idx),
                (SYM_POSITION_MAP, IonList([
                    IonStruct([(SYM_EID, IonEidRef(e)), (SYM_SECTION_ID, IonEidRef(s))])
                    for e, s in buckets[b_idx]
                ])),
            ]))
            self._record_blob(
                fragments, frag_props, gc_reachable,
                f"eidbucket_{b_idx}", "yj.eidhash_eid_section_map", bucket_ion,
            )

        pid_map_ion = IonAnnotation([SYM_SECTION_PID_COUNT_MAP], IonStruct([
            (SYM_POSITION_MAP, IonList([
                IonStruct([(SYM_SECTION_ID, IonEidRef(s)), (SYM_COUNT, c)])
                for s, c in section_pid_counts
            ])),
        ]))
        self._record_blob(
            fragments, frag_props, gc_reachable,
            "yj.section_pid_count_map", "yj.section_pid_count_map", pid_map_ion,
        )

    @staticmethod
    def _write_kdf_and_kpf(
        *,
        out_path: Path,
        page_data: list[dict],
        fragments: list[tuple[str, str, bytes]],
        frag_props: list[tuple[str, str, str]],
        gc_frag_props: list[tuple[str, str, str]],
        gc_reachable: set[str],
    ) -> None:
        """Write SQLite KDF container, apply fingerprints, compute book.kcb, and package ZIP."""
        with tempfile.TemporaryDirectory(prefix="m2k_kpf_") as tmp_dir_str:
            db_path = Path(tmp_dir_str) / "book.kdf"
            conn = sqlite3.connect(db_path)
            try:
                cur = conn.cursor()
                cur.execute(
                    "CREATE TABLE capabilities(key char(20), version smallint, "
                    "primary key (key, version)) without rowid"
                )
                cur.execute(
                    "CREATE TABLE fragments(id char(40), payload_type char(10), "
                    "payload_value blob, primary key (id))"
                )
                cur.execute(
                    "CREATE TABLE fragment_properties(id char(40), key char(40), "
                    "value char(40), primary key (id, key, value)) without rowid"
                )
                cur.execute(
                    "CREATE TABLE gc_fragment_properties(id varchar(40), key varchar(40), "
                    "value varchar(40), primary key (id, key, value)) without rowid"
                )
                cur.execute(
                    "CREATE TABLE gc_reachable(id varchar(40), primary key (id)) without rowid"
                )
                cur.execute("INSERT INTO capabilities VALUES ('db.schema', 1)")
                for fid, ptype, pval in fragments:
                    cur.execute("INSERT INTO fragments VALUES (?, ?, ?)", (fid, ptype, pval))
                for fid, k, v in set(frag_props):
                    cur.execute("INSERT OR IGNORE INTO fragment_properties VALUES (?, ?, ?)", (fid, k, v))
                for fid, k, v in set(gc_frag_props):
                    cur.execute("INSERT OR IGNORE INTO gc_fragment_properties VALUES (?, ?, ?)", (fid, k, v))
                for rid in sorted(gc_reachable):
                    cur.execute("INSERT OR IGNORE INTO gc_reachable VALUES (?)", (rid,))
                conn.commit()
            finally:
                conn.close()

            raw_db_bytes = db_path.read_bytes()
            wrapped_kdf = add_fingerprints(raw_db_bytes)

            hasher = hashlib.md5(raw_db_bytes)
            for pd in page_data:
                hasher.update(pd["img_path"].read_bytes())
            kcb_raw = json.dumps(
                {"content_hash": hasher.hexdigest(), "metadata": {"format": "yj"}},
                indent=2,
            ).encode("utf-8")

            with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("book.kdf", wrapped_kdf)
                zf.writestr("book.kcb", kcb_raw)
                for pd in page_data:
                    zf.write(pd["img_path"], f"res/{pd['rsrc_id']}")
