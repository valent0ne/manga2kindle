"""Amazon Ion 1.0 binary serialization for Kindle Package Format (KPF)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

ION_BVM: bytes = b"\xe0\x01\x00\xea"
YJ_SYMBOLS_MAX_ID: int = 851

# Ion system symbol IDs ($1..$9)
SYM_SYS_NAME: int = 4
SYM_SYS_VERSION: int = 5
SYM_SYS_IMPORTS: int = 6
SYM_SYS_SYMBOLS: int = 7
SYM_SYS_MAX_ID: int = 8

# YJ_symbols shared symbol table IDs (version 10, max_id=842)
SYM_FORMAT_VERSION: int = 16
SYM_WIDTH: int = 56
SYM_HEIGHT: int = 57
SYM_LEFT: int = 59
SYM_TOP: int = 58
SYM_PAGE_WIDTH: int = 66
SYM_PAGE_HEIGHT: int = 67
SYM_PAGE_TEMPLATE_TYPE: int = 140
SYM_SECTION_CONTENT: int = 141
SYM_COUNT: int = 144
SYM_CHILDREN: int = 146
SYM_LAYOUT_TYPE: int = 156
SYM_NODE_TYPE: int = 159
SYM_FORMAT: int = 161
SYM_TARGET_ID: int = 163
SYM_EXTERNAL_RESOURCE: int = 164
SYM_LOCATION: int = 165
SYM_READING_ORDERS: int = 169
SYM_SECTIONS: int = 170
SYM_SECTION_ID: int = 174
SYM_RESOURCE_ID: int = 175
SYM_STORYLINE_ID: int = 176
SYM_READING_ORDER_NAME: int = 178
SYM_POSITION_MAP: int = 181
SYM_FIT_TYPE: int = 183
SYM_EID: int = 185
SYM_LAYOUT: int = 192
SYM_METADATA: int = 258
SYM_STORYLINE: int = 259
SYM_SECTION: int = 260
SYM_CONTAINER: int = 270
SYM_LEAF: int = 271
SYM_PNG: int = 284
SYM_JPG: int = 285
SYM_UNIT: int = 306
SYM_VALUE: int = 307
SYM_PERCENTAGE: int = 314
SYM_FIXED: int = 320
SYM_BLOCK: int = 323
SYM_FIT_BOTH: int = 324
SYM_FIXED_LAYOUT: int = 326
SYM_DEFAULT: int = 351
SYM_FIXED_LAYOUT_MODE_RTL: int = 375
SYM_FIXED_LAYOUT_MODE_LTR: int = 376
SYM_ABSOLUTE: int = 377
SYM_IMAGE_WIDTH: int = 422
SYM_IMAGE_HEIGHT: int = 423
SYM_ACTIONS: int = 426
SYM_ORDINAL: int = 427
SYM_ACTION: int = 428
SYM_RIGHT_TO_LEFT_BINDING: int = 441
SYM_LEFT_TO_RIGHT_BINDING: int = 442
SYM_MAGNIFY: int = 468
SYM_SOURCE_ID: int = 474
SYM_BOOK_METADATA: int = 490
SYM_CATEGORIES: int = 491
SYM_KEY: int = 492
SYM_CATEGORY_NAME: int = 495
SYM_DOCUMENT_DATA: int = 538
SYM_POSITION_TYPE: int = 546
SYM_RIGHT_TO_LEFT_PAGE: int = 557
SYM_LEFT_TO_RIGHT_PAGE: int = 558
SYM_PAGE_PROGRESSION_DIR: int = 560
SYM_BINDING: int = 581
SYM_CONTENT_FEATURES: int = 585
SYM_NAMESPACE: int = 586
SYM_MAJOR_VERSION: int = 587
SYM_MINOR_VERSION: int = 588
SYM_PROPERTIES: int = 589
SYM_FEATURES_LIST: int = 590
SYM_AUXILIARY_DATA: int = 597
SYM_SELF_REF: int = 598
SYM_BUCKET_INDEX: int = 602
SYM_STRUCTURE: int = 608
SYM_SECTION_POSITION_ID_MAP: int = 609
SYM_EIDHASH_EID_SECTION_MAP: int = 610
SYM_SECTION_PID_COUNT_MAP: int = 611
SYM_RESOURCE_LIST_REF: int = 613
SYM_PAN_ZOOM_VIEWER: int = 684

# Local symbol IDs
SYM_LOCAL_SOURCE_FILE_NAME: int = 852
SYM_LOCAL_ORIGINAL_RESOURCE: int = 853
SYM_LOCAL_PRESERVED_ORIGINAL: int = 854


@dataclass(frozen=True)
class IonSymbol:
    """Represents an Amazon Ion symbol identifier (type 7)."""

    sid: int


@dataclass(frozen=True)
class IonAnnotation:
    """Wraps an Ion value with one or more symbol ID annotations (type 14)."""

    sids: list[int]
    value: Any


@dataclass(frozen=True)
class IonStruct:
    """Represents an Ion struct (type 13) of (field_sid, value) tuples."""

    fields: list[tuple[int, Any]]


@dataclass(frozen=True)
class IonList:
    """Represents an Ion list (type 11)."""

    values: list[Any]


@dataclass(frozen=True)
class IonSExp:
    """Represents an Ion S-expression (type 12)."""

    values: list[Any]


@dataclass(frozen=True)
class IonFloat64:
    """Represents an explicit 64-bit float (IEEE 754 double, type 4)."""

    value: float


@dataclass(frozen=True)
class IonEidRef:
    """Annotated string reference $598::'eid_string' used in KDF schema."""

    eid: str


class IonBinaryWriter:
    """Encodes Python structures and Ion models into Amazon Ion 1.0 binary format."""

    @staticmethod
    def encode_varuint(value: int) -> bytes:
        """Encode an unsigned integer as an Ion VarUInt (MSB=1 on final byte)."""
        if value < 0:
            raise ValueError("VarUInt cannot encode negative values")
        groups: list[int] = []
        if value == 0:
            groups.append(0)
        else:
            v = value
            while v > 0:
                groups.append(v & 0x7F)
                v >>= 7
            groups.reverse()
        result = bytearray()
        for i, g in enumerate(groups):
            if i == len(groups) - 1:
                result.append(g | 0x80)
            else:
                result.append(g)
        return bytes(result)

    @classmethod
    def _ion_type_descriptor(cls, type_id: int, length: int) -> bytes:
        """Encode an Ion type descriptor byte plus length VarUInt if length >= 14."""
        if length < 14:
            return bytes([(type_id << 4) | length])
        return bytes([(type_id << 4) | 14]) + cls.encode_varuint(length)

    @classmethod
    def encode_int(cls, val: int) -> bytes:
        """Encode integer (type 2 for positive/zero, type 3 for negative)."""
        if val == 0:
            return b"\x20"
        negative = val < 0
        abs_val = abs(val)
        data = bytearray()
        v = abs_val
        while v > 0:
            data.append(v & 0xFF)
            v >>= 8
        data.reverse()
        type_id = 3 if negative else 2
        return cls._ion_type_descriptor(type_id, len(data)) + bytes(data)

    @classmethod
    def encode_float64(cls, val: float) -> bytes:
        """Encode 64-bit float (IEEE 754 double, type 4)."""
        return b"\x48" + struct.pack(">d", val)

    @classmethod
    def encode_string(cls, s: str) -> bytes:
        """Encode UTF-8 string (type 8)."""
        encoded = s.encode("utf-8")
        return cls._ion_type_descriptor(8, len(encoded)) + encoded

    @classmethod
    def encode_symbol(cls, sid: int) -> bytes:
        """Encode an Ion symbol ID (type 7)."""
        if sid == 0:
            return b"\x70"
        data = bytearray()
        v = sid
        while v > 0:
            data.append(v & 0xFF)
            v >>= 8
        data.reverse()
        return cls._ion_type_descriptor(7, len(data)) + bytes(data)

    @classmethod
    def encode_eid_ref(cls, eid: str) -> bytes:
        """Encode $598::'eid' annotation reference."""
        return cls.encode_annotation([SYM_SELF_REF], cls.encode_string(eid))

    @classmethod
    def encode_list(cls, items: list[Any]) -> bytes:
        """Encode Ion list (type 11)."""
        body = b"".join(cls.encode(item, include_bvm=False) for item in items)
        return cls._ion_type_descriptor(11, len(body)) + body

    @classmethod
    def encode_sexp(cls, items: list[Any]) -> bytes:
        """Encode Ion S-expression (type 12)."""
        body = b"".join(cls.encode(item, include_bvm=False) for item in items)
        return cls._ion_type_descriptor(12, len(body)) + body

    @classmethod
    def encode_struct(cls, fields: list[tuple[int, Any]]) -> bytes:
        """Encode Ion struct (type 13) from (field_sid, value) tuples."""
        body = b"".join(
            cls.encode_varuint(sid) + cls.encode(val, include_bvm=False)
            for sid, val in fields
        )
        return cls._ion_type_descriptor(13, len(body)) + body

    @classmethod
    def encode_annotation(cls, annotation_sids: list[int], value: Any) -> bytes:
        """Wrap an Ion value with one or more symbol ID annotations (type 14)."""
        val_bytes = cls.encode(value, include_bvm=False)
        annot_bytes = b"".join(cls.encode_varuint(sid) for sid in annotation_sids)
        annot_len = cls.encode_varuint(len(annot_bytes))
        inner = annot_len + annot_bytes + val_bytes
        return cls._ion_type_descriptor(14, len(inner)) + inner

    @classmethod
    def encode(cls, value: Any, include_bvm: bool = True) -> bytes:
        """Encode an arbitrary Ion value or Python primitive."""
        if value is None:
            raw = b"\x0f"
        elif isinstance(value, bool):
            raw = bytes([0x11 if value else 0x10])
        elif isinstance(value, int):
            raw = cls.encode_int(value)
        elif isinstance(value, float):
            raw = cls.encode_float64(value)
        elif isinstance(value, str):
            raw = cls.encode_string(value)
        elif isinstance(value, IonFloat64):
            raw = cls.encode_float64(value.value)
        elif isinstance(value, IonSymbol):
            raw = cls.encode_symbol(value.sid)
        elif isinstance(value, IonEidRef):
            raw = cls.encode_eid_ref(value.eid)
        elif isinstance(value, IonList):
            raw = cls.encode_list(value.values)
        elif isinstance(value, IonSExp):
            raw = cls.encode_sexp(value.values)
        elif isinstance(value, IonStruct):
            raw = cls.encode_struct(value.fields)
        elif isinstance(value, IonAnnotation):
            raw = cls.encode_annotation(value.sids, value.value)
        elif isinstance(value, (bytes, bytearray)):
            raw = bytes(value)
        else:
            raise TypeError(f"Unsupported Ion value type: {type(value)}")

        if include_bvm:
            if raw.startswith(ION_BVM):
                return raw
            return ION_BVM + raw
        return raw

    @classmethod
    def build_ion_symbol_table(cls, local_symbols: list[str] | None = None) -> bytes:
        """Build the standard $ion_symbol_table importing YJ_symbols v10."""
        syms = list(local_symbols) if local_symbols is not None else [
            "yj.authoring.source_file_name",
            "yj.authoring.original_resource",
            "yj.authoring.preserved_original_resource",
        ]
        max_sid = YJ_SYMBOLS_MAX_ID + len(syms)
        import_entry = IonStruct([
            (SYM_SYS_NAME, "YJ_symbols"),
            (SYM_SYS_VERSION, 10),
            (SYM_SYS_MAX_ID, 842),
        ])
        fields: list[tuple[int, Any]] = [
            (SYM_SYS_MAX_ID, max_sid),
            (SYM_SYS_IMPORTS, IonList([import_entry])),
            (SYM_SYS_SYMBOLS, IonList([cls.encode_string(s) for s in syms])),
        ]
        return cls.encode(IonAnnotation([3], IonStruct(fields)), include_bvm=True)
