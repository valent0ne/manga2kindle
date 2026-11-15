"""Fixed-layout KPF package builder with Guided View panel magnification."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import uuid
import zipfile
from pathlib import Path
from PIL import Image

from manga2kindle.models import (
    ConversionConfig,
    PageAsset,
    PanelBox,
    ReadingDirection,
    ZoomRect,
)
from manga2kindle.packaging.ion import (
    ION_BVM,
    SYM_ABSOLUTE,
    SYM_ACTION,
    SYM_ACTIONS,
    SYM_AUXILIARY_DATA,
    SYM_BINDING,
    SYM_BLOCK,
    SYM_BOOK_METADATA,
    SYM_BUCKET_INDEX,
    SYM_CATEGORIES,
    SYM_CATEGORY_NAME,
    SYM_CHILDREN,
    SYM_CONTAINER,
    SYM_CONTENT_FEATURES,
    SYM_COUNT,
    SYM_DEFAULT,
    SYM_DOCUMENT_DATA,
    SYM_EID,
    SYM_EIDHASH_EID_SECTION_MAP,
    SYM_EXTERNAL_RESOURCE,
    SYM_FEATURES_LIST,
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
    SYM_PAN_ZOOM_VIEWER,
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


class KpfBuilder:
    """Fixed-layout KPF package builder with Guided View panel magnification."""

    @staticmethod
    def compute_zoom_rect(
        panel: PanelBox,
        page_w: int,
        page_h: int,
        zoom_factor: float = 1.5,
    ) -> ZoomRect:
        """Compute zoomed and clamped bounding box for Guided View magnification."""
        factor = max(1.0, float(zoom_factor))
        scaled_w = int(round(panel.w * factor))
        scaled_h = int(round(panel.h * factor))

        target_w = max(1, min(scaled_w, page_w))
        target_h = max(1, min(scaled_h, page_h))

        cx = panel.x + panel.w / 2.0
        cy = panel.y + panel.h / 2.0

        target_x = int(round(cx - target_w / 2.0))
        target_y = int(round(cy - target_h / 2.0))

        clamped_x = max(0, min(target_x, page_w - target_w))
        clamped_y = max(0, min(target_y, page_h - target_h))

        return ZoomRect(x=clamped_x, y=clamped_y, w=target_w, h=target_h)

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

        # Prepare per-page metadata and allocate sequential IDs
        page_data: list[dict] = []
        for page in pages:
            img_path = page.processed_path or page.source_path
            w = page.width
            h = page.height
            if w <= 0 or h <= 0:
                with Image.open(img_path) as im:
                    w, h = im.size

            # Format detection
            with open(img_path, "rb") as f:
                header = f.read(8)
            is_png = header.startswith(b"\x89PNG\r\n\x1a\n")
            fmt_sym = SYM_PNG if is_png else SYM_JPG

            sid = ids.next_id("s")
            t_eid = ids.next_id("c")
            l_id = ids.next_id("l")
            i_cont = ids.next_id("i")
            i_leaf = ids.next_id("i")
            e_id = ids.next_id("e")
            rsrc_id = ids.next_id("rsrc")
            d_id = ids.next_id("d")

            panels_info: list[dict] = []
            for idx, p in enumerate(page.panels):
                panels_info.append({
                    "panel": p,
                    "ordinal": p.ordinal if p.ordinal > 0 else (idx + 1),
                    "src_eid": ids.next_id("p"),
                    "tgt_eid": ids.next_id("t"),
                })

            page_data.append({
                "page": page,
                "img_path": img_path,
                "width": w,
                "height": h,
                "fmt_sym": fmt_sym,
                "sid": sid,
                "t_eid": t_eid,
                "l_id": l_id,
                "i_cont": i_cont,
                "i_leaf": i_leaf,
                "e_id": e_id,
                "rsrc_id": rsrc_id,
                "d_id": d_id,
                "panels": panels_info,
            })

        # Identify cover image resource EID
        cover_page = next((pd for pd in page_data if pd["page"].is_cover), page_data[0])
        cover_resource_eid = cover_page["e_id"]

        fragments: list[tuple[str, str, bytes]] = []
        frag_props: list[tuple[str, str, str]] = []
        gc_frag_props: list[tuple[str, str, str]] = []
        gc_reachable: set[str] = set()

        # 1. $ion_symbol_table
        fragments.append((
            "$ion_symbol_table",
            "blob",
            IonBinaryWriter.build_ion_symbol_table(),
        ))
        frag_props.append(("$ion_symbol_table", "element_type", "$ion_symbol_table"))
        gc_reachable.add("$ion_symbol_table")

        # 2. max_id
        fragments.append((
            "max_id",
            "blob",
            IonBinaryWriter.encode(YJ_SYMBOLS_MAX_ID + 3, include_bvm=True),
        ))
        frag_props.append(("max_id", "element_type", "max_id"))
        gc_reachable.add("max_id")

        # 3. book_navigation
        fragments.append(("book_navigation", "blob", ION_BVM))
        frag_props.append(("book_navigation", "element_type", "book_navigation"))
        gc_frag_props.append(("book_navigation", "child", "book_navigation"))
        gc_reachable.add("book_navigation")

        # 4. book_metadata
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
            (SYM_METADATA, IonList([
                IonStruct([(SYM_KEY, "selection"), (SYM_VALUE, "enabled")]),
            ])),
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
        fragments.append(("book_metadata", "blob", IonBinaryWriter.encode(bm_ion, include_bvm=True)))
        frag_props.append(("book_metadata", "element_type", "book_metadata"))
        gc_reachable.add("book_metadata")

        # 5. content_features
        cf_fl = IonStruct([
            (SYM_NAMESPACE, "com.amazon.yjconversion"),
            (SYM_KEY, "yj_non_pdf_fixed_layout"),
            (SYM_PROPERTIES, IonStruct([
                (SYM_SYS_VERSION, IonStruct([
                    (SYM_MAJOR_VERSION, 2),
                    (SYM_MINOR_VERSION, 0),
                ])),
            ])),
        ])
        cf_pp = IonStruct([
            (SYM_NAMESPACE, "com.amazon.yjconversion"),
            (SYM_KEY, "yj_publisher_panels"),
            (SYM_PROPERTIES, IonStruct([
                (SYM_SYS_VERSION, IonStruct([
                    (SYM_MAJOR_VERSION, 2),
                    (SYM_MINOR_VERSION, 0),
                ])),
            ])),
        ])
        cf_ion = IonAnnotation([SYM_CONTENT_FEATURES], IonStruct([
            (SYM_SELF_REF, IonSymbol(SYM_CONTENT_FEATURES)),
            (SYM_FEATURES_LIST, IonList([cf_fl, cf_pp])),
        ]))
        fragments.append(("content_features", "blob", IonBinaryWriter.encode(cf_ion, include_bvm=True)))
        frag_props.append(("content_features", "element_type", "content_features"))
        gc_reachable.add("content_features")

        # 6. document_data & metadata
        section_ids = [pd["sid"] for pd in page_data]
        is_rtl = config.direction == ReadingDirection.RTL
        layout_mode_sym = SYM_FIXED_LAYOUT_MODE_RTL if is_rtl else SYM_FIXED_LAYOUT_MODE_LTR
        page_dir_sym = SYM_RIGHT_TO_LEFT_PAGE if is_rtl else SYM_LEFT_TO_RIGHT_PAGE
        binding_sym = SYM_RIGHT_TO_LEFT_BINDING if is_rtl else SYM_LEFT_TO_RIGHT_BINDING

        reading_order = IonStruct([
            (SYM_READING_ORDER_NAME, IonSymbol(SYM_DEFAULT)),
            (SYM_SECTIONS, IonList([IonEidRef(s) for s in section_ids])),
        ])
        aux_ref_struct = IonStruct([
            (SYM_RESOURCE_LIST_REF, IonEidRef(res_list_id)),
        ])

        doc_data_ion = IonAnnotation([SYM_DOCUMENT_DATA], IonStruct([
            (SYM_FORMAT_VERSION, IonFloat64(16.0)),
            (SYM_PAGE_PROGRESSION_DIR, IonSymbol(page_dir_sym)),
            (SYM_SYS_MAX_ID, ids.total_count),
            (SYM_LAYOUT, IonSymbol(layout_mode_sym)),
            (SYM_BINDING, IonSymbol(binding_sym)),
            (SYM_AUXILIARY_DATA, aux_ref_struct),
            (SYM_READING_ORDERS, IonList([reading_order])),
        ]))
        fragments.append(("document_data", "blob", IonBinaryWriter.encode(doc_data_ion, include_bvm=True)))
        frag_props.append(("document_data", "element_type", "document_data"))
        frag_props.append(("document_data", "child", res_list_id))
        gc_frag_props.append(("document_data", "child", res_list_id))
        gc_reachable.add("document_data")

        metadata_ion = IonAnnotation([SYM_METADATA], IonStruct([
            (SYM_READING_ORDERS, IonList([reading_order])),
        ]))
        fragments.append(("metadata", "blob", IonBinaryWriter.encode(metadata_ion, include_bvm=True)))
        frag_props.append(("metadata", "element_type", "metadata"))
        gc_reachable.add("metadata")

        # 7. Resource list auxiliary data
        all_d_ids = [pd["d_id"] for pd in page_data]
        res_list_ion = IonAnnotation([SYM_AUXILIARY_DATA], IonStruct([
            (SYM_SELF_REF, IonEidRef(res_list_id)),
            (SYM_METADATA, IonList([
                IonStruct([
                    (SYM_KEY, "auxData_resource_list"),
                    (SYM_VALUE, IonList([IonEidRef(d) for d in all_d_ids])),
                ]),
            ])),
        ]))
        fragments.append((res_list_id, "blob", IonBinaryWriter.encode(res_list_ion, include_bvm=True)))
        frag_props.append((res_list_id, "element_type", "auxiliary_data"))
        gc_reachable.add(res_list_id)

        # 8. Per-page sections, storylines, images, and panels
        eid_section_map: list[tuple[str, str]] = []
        section_pid_counts: list[tuple[str, int]] = []

        for pd in page_data:
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

            # Section fragment
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
            fragments.append((sid, "blob", IonBinaryWriter.encode(section_ion, include_bvm=True)))
            frag_props.append((sid, "element_type", "section"))
            frag_props.append((sid, "child", f"{sid}-ad"))
            frag_props.append((sid, "child", l_id))
            gc_reachable.add(sid)
            gc_reachable.add(f"{sid}-ad")

            # Storyline
            story_children = [i_cont] + [p["src_eid"] for p in panels] + [p["tgt_eid"] for p in panels]
            story_ion = IonAnnotation([SYM_STORYLINE], IonStruct([
                (SYM_STORYLINE_ID, IonEidRef(l_id)),
                (SYM_CHILDREN, IonList([IonEidRef(cid) for cid in story_children])),
            ]))
            fragments.append((l_id, "blob", IonBinaryWriter.encode(story_ion, include_bvm=True)))
            frag_props.append((l_id, "element_type", "storyline"))
            for cid in story_children:
                frag_props.append((l_id, "child", cid))
                gc_frag_props.append((l_id, "child", cid))
            frag_props.append((l_id, "child", l_id))
            gc_frag_props.append((l_id, "child", l_id))
            gc_reachable.add(l_id)

            # Image container + leaf
            cont_ion = IonAnnotation([SYM_STRUCTURE], IonStruct([
                (SYM_SELF_REF, IonEidRef(i_cont)),
                (SYM_WIDTH, w),
                (SYM_HEIGHT, h),
                (SYM_POSITION_TYPE, IonSymbol(SYM_ABSOLUTE)),
                (SYM_LAYOUT_TYPE, IonSymbol(SYM_BLOCK)),
                (SYM_NODE_TYPE, IonSymbol(SYM_CONTAINER)),
                (SYM_CHILDREN, IonList([IonEidRef(i_leaf)])),
            ]))
            fragments.append((i_cont, "blob", IonBinaryWriter.encode(cont_ion, include_bvm=True)))
            frag_props.append((i_cont, "element_type", "structure"))
            frag_props.append((i_cont, "child", i_leaf))
            gc_reachable.add(i_cont)

            leaf_ion = IonAnnotation([SYM_STRUCTURE], IonStruct([
                (SYM_SELF_REF, IonEidRef(i_leaf)),
                (SYM_WIDTH, w),
                (SYM_HEIGHT, h),
                (SYM_RESOURCE_ID, IonEidRef(e_id)),
                (SYM_POSITION_TYPE, IonSymbol(SYM_ABSOLUTE)),
                (SYM_NODE_TYPE, IonSymbol(SYM_LEAF)),
                (SYM_FIT_TYPE, IonSymbol(SYM_FIT_BOTH)),
            ]))
            fragments.append((i_leaf, "blob", IonBinaryWriter.encode(leaf_ion, include_bvm=True)))
            frag_props.append((i_leaf, "element_type", "structure"))
            frag_props.append((i_leaf, "child", e_id))
            gc_reachable.add(i_leaf)

            # Guided View Panel Containers
            for p_info in panels:
                p_box: PanelBox = p_info["panel"]
                ordinal: int = p_info["ordinal"]
                src_eid: str = p_info["src_eid"]
                tgt_eid: str = p_info["tgt_eid"]

                zoom_rect = self.compute_zoom_rect(p_box, w, h, config.zoom_factor)

                activate_act = IonStruct([
                    (SYM_ACTION, IonSymbol(SYM_MAGNIFY)),
                    (SYM_SOURCE_ID, IonEidRef(src_eid)),
                    (SYM_TARGET_ID, IonEidRef(tgt_eid)),
                ])

                src_node = IonAnnotation([SYM_STRUCTURE], IonStruct([
                    (SYM_SELF_REF, IonEidRef(src_eid)),
                    (SYM_NODE_TYPE, IonSymbol(SYM_CONTAINER)),
                    (SYM_LAYOUT_TYPE, IonSymbol(SYM_FIT_BOTH)),
                    (SYM_FIT_TYPE, IonSymbol(SYM_FIT_BOTH)),
                    (SYM_LEFT, _dim_pct(p_box.x, w)),
                    (SYM_TOP, _dim_pct(p_box.y, h)),
                    (SYM_WIDTH, _dim_pct(p_box.w, w)),
                    (SYM_HEIGHT, _dim_pct(p_box.h, h)),
                    (SYM_PAN_ZOOM_VIEWER, IonSymbol(SYM_RIGHT_TO_LEFT_BINDING)),
                    (SYM_ORDINAL, ordinal),
                    (SYM_ACTIONS, IonList([activate_act])),
                ]))
                fragments.append((src_eid, "blob", IonBinaryWriter.encode(src_node, include_bvm=True)))
                frag_props.append((src_eid, "element_type", "structure"))
                gc_reachable.add(src_eid)

                tgt_node = IonAnnotation([SYM_STRUCTURE], IonStruct([
                    (SYM_SELF_REF, IonEidRef(tgt_eid)),
                    (SYM_NODE_TYPE, IonSymbol(SYM_CONTAINER)),
                    (SYM_LAYOUT_TYPE, IonSymbol(SYM_FIT_BOTH)),
                    (SYM_FIT_TYPE, IonSymbol(SYM_FIT_BOTH)),
                    (SYM_LEFT, _dim_pct(zoom_rect.x, w)),
                    (SYM_TOP, _dim_pct(zoom_rect.y, h)),
                    (SYM_WIDTH, _dim_pct(zoom_rect.w, w)),
                    (SYM_HEIGHT, _dim_pct(zoom_rect.h, h)),
                ]))
                fragments.append((tgt_eid, "blob", IonBinaryWriter.encode(tgt_node, include_bvm=True)))
                frag_props.append((tgt_eid, "element_type", "structure"))
                gc_reachable.add(tgt_eid)

            # External resource
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
            fragments.append((e_id, "blob", IonBinaryWriter.encode(ext_res_ion, include_bvm=True)))
            frag_props.append((e_id, "element_type", "external_resource"))
            frag_props.append((e_id, "child", d_id))
            frag_props.append((e_id, "child", rsrc_id))
            gc_reachable.add(e_id)

            # Raw media fragment (bcRawMedia path stored as UTF-8 bytes)
            fragments.append((rsrc_id, "path", rsrc_rel_path.encode("utf-8")))
            frag_props.append((rsrc_id, "element_type", "bcRawMedia"))
            gc_reachable.add(rsrc_id)

            # Auxiliary data for resource
            img_file_size = pd["img_path"].stat().st_size
            img_mtime = int(pd["img_path"].stat().st_mtime)
            aux_ion = IonAnnotation([SYM_AUXILIARY_DATA], IonStruct([
                (SYM_SELF_REF, IonEidRef(d_id)),
                (SYM_METADATA, IonList([
                    IonStruct([(SYM_KEY, "type"), (SYM_VALUE, "resource")]),
                    IonStruct([(SYM_KEY, "resource_stream"), (SYM_VALUE, rsrc_rel_path)]),
                    IonStruct([(SYM_KEY, "size"), (SYM_VALUE, str(img_file_size))]),
                    IonStruct([(SYM_KEY, "modified_time"), (SYM_VALUE, str(img_mtime))]),
                    IonStruct([(SYM_KEY, "location"), (SYM_VALUE, str(pd["img_path"]))]),
                ])),
            ]))
            fragments.append((d_id, "blob", IonBinaryWriter.encode(aux_ion, include_bvm=True)))
            frag_props.append((d_id, "element_type", "auxiliary_data"))
            gc_reachable.add(d_id)

            # Section position ID map (<sid>-spm)
            spim_eids = [t_eid, i_cont, i_leaf] + [p["src_eid"] for p in panels] + [p["tgt_eid"] for p in panels]
            spm_ion = IonAnnotation([SYM_SECTION_POSITION_ID_MAP], IonStruct([
                (SYM_SECTION_ID, IonEidRef(sid)),
                (SYM_POSITION_MAP, IonList([
                    IonList([idx + 1, IonEidRef(e)]) for idx, e in enumerate(spim_eids)
                ])),
            ]))
            spm_id = f"{sid}-spm"
            fragments.append((spm_id, "blob", IonBinaryWriter.encode(spm_ion, include_bvm=True)))
            frag_props.append((spm_id, "element_type", "section_position_id_map"))
            gc_reachable.add(spm_id)

            # Map entries for EID hash buckets
            eid_section_map.extend([
                (sid, sid),
                (t_eid, sid),
                (i_cont, sid),
                (i_leaf, sid),
            ])
            for p_info in panels:
                eid_section_map.append((p_info["src_eid"], sid))
                eid_section_map.append((p_info["tgt_eid"], sid))

            section_pid_counts.append((sid, len(spim_eids)))

        # 9. EID hash buckets
        num_buckets = 1 if len(eid_section_map) <= 10 else max(1, round(len(eid_section_map) / 10.3))
        buckets: dict[int, list[tuple[str, str]]] = {i: [] for i in range(num_buckets)}
        for eid_str, sec_id in eid_section_map:
            bucket_idx = sum(ord(c) for c in eid_str) % num_buckets
            buckets[bucket_idx].append((eid_str, sec_id))

        for b_idx in range(num_buckets):
            bucket_id = f"eidbucket_{b_idx}"
            b_entries = [
                IonStruct([
                    (SYM_EID, IonEidRef(e)),
                    (SYM_SECTION_ID, IonEidRef(s)),
                ])
                for e, s in buckets[b_idx]
            ]
            bucket_ion = IonAnnotation([SYM_EIDHASH_EID_SECTION_MAP], IonStruct([
                (SYM_BUCKET_INDEX, b_idx),
                (SYM_POSITION_MAP, IonList(b_entries)),
            ]))
            fragments.append((bucket_id, "blob", IonBinaryWriter.encode(bucket_ion, include_bvm=True)))
            frag_props.append((bucket_id, "element_type", "yj.eidhash_eid_section_map"))
            gc_reachable.add(bucket_id)

        # 10. Section PID count map
        pid_map_ion = IonAnnotation([SYM_SECTION_PID_COUNT_MAP], IonStruct([
            (SYM_POSITION_MAP, IonList([
                IonStruct([(SYM_SECTION_ID, IonEidRef(s)), (SYM_COUNT, c)])
                for s, c in section_pid_counts
            ])),
        ]))
        fragments.append((
            "yj.section_pid_count_map",
            "blob",
            IonBinaryWriter.encode(pid_map_ion, include_bvm=True),
        ))
        frag_props.append(("yj.section_pid_count_map", "element_type", "yj.section_pid_count_map"))
        gc_reachable.add("yj.section_pid_count_map")

        # 11. Write SQLite database and wrap with fingerprints
        with tempfile.TemporaryDirectory(prefix="m2k_kpf_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            db_path = tmp_dir / "book.kdf"

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

            # Compute MD5 content hash from raw DB and images
            hasher = hashlib.md5(raw_db_bytes)
            for pd in page_data:
                hasher.update(pd["img_path"].read_bytes())
            content_hash_hex = hasher.hexdigest()

            kcb_obj = {
                "content_hash": content_hash_hex,
                "metadata": {
                    "format": "yj",
                },
            }
            kcb_raw = json.dumps(kcb_obj, indent=2).encode("utf-8")

            # Write output ZIP archive
            with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("book.kdf", wrapped_kdf)
                zf.writestr("book.kcb", kcb_raw)
                for pd in page_data:
                    zf.write(pd["img_path"], f"res/{pd['rsrc_id']}")

        return out_path
