"""
Utilities for reading and editing <box> elements inside an annotations XML like:

<annotations>
  <image id="0" name="00002_Q_page_1.tif" width="2477" height="3504">
    <box label="censor" xtl="280.80" ytl="511.70" xbr="2209.40" ybr="842.20" />
    <box label="roi" ...>
      <attribute name="blank">false</attribute>
    </box>
  </image>
  ...
</annotations>

This module provides helpers to:
- Load and save the XML
- Query images and boxes (filter by image id/name and box label)
- Read attributes (e.g., <attribute name="blank">true/false</attribute>)
- Add/update attributes on selected <box> elements

Compatible with Python 3.9+ and the standard library (xml.etree.ElementTree).
"""

from __future__ import annotations
'''
from __future__ import annotations tells Python to delay 
evaluating type hints so they can safely reference classes or types that haven’t been defined yet.'''
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import xml.etree.ElementTree as ET

# ---------------------------
# Data models
# ---------------------------

@dataclass
class ImageInfo:
    id: str
    name: str
    width: Optional[int] = None
    height: Optional[int] = None
    element: Optional[ET.Element] = None  # underlying XML element (optional)

@dataclass
class Box:
    label: str
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    z_order: Optional[int] = None
    source: Optional[str] = None
    occluded: Optional[str] = None
    attributes: Dict[str, str] = None
    image: Optional[ImageInfo] = None
    element: Optional[ET.Element] = None  # underlying XML element (optional)

    @property
    def width(self) -> float:
        return self.xbr - self.xtl

    @property
    def height(self) -> float:
        return self.ybr - self.ytl

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

# ---------------------------
# Parsing & IO
# ---------------------------

def load_xml(path: str) -> Tuple[ET.ElementTree, ET.Element]:
    """Load XML from disk and return (tree, root)."""
    tree = ET.parse(path)
    return tree, tree.getroot()


def loads_xml(xml_text: str) -> Tuple[ET.ElementTree, ET.Element]:
    """Parse XML from a string and return (tree, root)."""
    root = ET.fromstring(xml_text)
    tree = ET.ElementTree(root)
    return tree, root


def save_xml(tree: ET.ElementTree, path: str) -> None:
    """Write XML back to disk with UTF-8 encoding and XML declaration."""
    tree.write(path, encoding="utf-8", xml_declaration=True)

# ---------------------------
# Queries (images & boxes)
# ---------------------------

def iter_images(root: ET.Element) -> Iterator[ImageInfo]:
    """Yield ImageInfo for each <image> element under <annotations>."""
    for img in root.findall(".//image"):
        yield ImageInfo(
            id=img.get("id", ""),
            name=img.get("name", ""),
            width=int(img.get("width")) if img.get("width") else None,
            height=int(img.get("height")) if img.get("height") else None,
            element=img,
        )


def find_image(root: ET.Element, *, image_id: Optional[str] = None, image_name: Optional[str] = None) -> Optional[ImageInfo]:
    """Find a single image by id or name (exact match). Returns None if not found.
    If both provided, both must match the same element.
    """
    for info in iter_images(root):
        if image_id is not None and info.id != str(image_id):
            continue
        if image_name is not None and info.name != image_name:
            continue
        return info
    return None


def _parse_box(box_el: ET.Element, img_info: Optional[ImageInfo]) -> Box:
    def f(attr: str, default: float = 0.0) -> float:
        v = box_el.get(attr)
        return float(v) if v is not None else default

    attrs: Dict[str, str] = {}
    for a in box_el.findall("./attribute"):
        name = a.get("name")
        if name is not None and a.text is not None:
            attrs[name] = a.text

    z_ord = box_el.get("z_order")
    return Box(
        label=box_el.get("label", ""),
        xtl=f("xtl"),
        ytl=f("ytl"),
        xbr=f("xbr"),
        ybr=f("ybr"),
        z_order=int(z_ord) if z_ord is not None else None,
        source=box_el.get("source"),
        occluded=box_el.get("occluded"),
        attributes=attrs,
        image=img_info,
        element=box_el,
    )


def iter_boxes(
    root: ET.Element,
    *,
    image_id: Optional[str] = None,
    image_name: Optional[str] = None,
    labels: Optional[Iterable[str]] = None,
) -> Iterator[Box]:
    """Yield boxes, optionally filtering by image id/name and box label(s).

    Example:
        for b in iter_boxes(root, image_id="0", labels=["roi"]):
            print(b.attributes.get("blank"))
    """
    label_set = set(labels) if labels else None
    for img_info in iter_images(root):
        if image_id is not None and img_info.id != str(image_id):
            continue
        if image_name is not None and img_info.name != image_name:
            continue
        for box_el in img_info.element.findall("./box"):
            if label_set and box_el.get("label") not in label_set:
                continue
            yield _parse_box(box_el, img_info)


# ---------------------------
# Attribute helpers
# ---------------------------

def get_box_attribute(box_el: ET.Element, name: str) -> Optional[str]:
    """Return the text value of <attribute name=...> inside a <box>, or None."""
    for a in box_el.findall("./attribute"):
        if a.get("name") == name:
            return a.text
    return None


def set_box_attribute(
    box_el: ET.Element,
    name: str,
    value: str,
    *,
    overwrite: bool = True,
) -> None:
    """Create or update an <attribute> inside a <box>.

    - If overwrite=False and the attribute exists, it is left unchanged.
    - Value is stored as text; pass strings like "true"/"false" or numbers as needed.
    """
    for a in box_el.findall("./attribute"):
        if a.get("name") == name:
            if overwrite:
                a.text = str(value)
            return
    a = ET.SubElement(box_el, "attribute", {"name": name})
    a.text = str(value)


def add_attribute_to_boxes(
    root: ET.Element,
    *,
    attr_name: str,
    attr_value: str,
    image_id: Optional[str] = None,
    image_name: Optional[str] = None,
    labels: Optional[Iterable[str]] = None,
    overwrite: bool = True,
) -> int:
    """Add/update an attribute on all selected boxes.

    Returns the number of boxes affected.
    """
    count = 0
    for box in iter_boxes(root, image_id=image_id, image_name=image_name, labels=labels):
        set_box_attribute(box.element, attr_name, attr_value, overwrite=overwrite)
        count += 1
    return count


# ---------------------------
# Convenience: common queries
# ---------------------------

def list_image_summaries(root: ET.Element) -> List[Dict[str, object]]:
    """Return a high-level summary for each image: counts per label and ROIs with blank attr."""
    summaries: List[Dict[str, object]] = []
    for img in iter_images(root):
        label_counts: Dict[str, int] = {}
        roi_blanks: List[str] = []
        for b in iter_boxes(root, image_id=img.id):
            label_counts[b.label] = label_counts.get(b.label, 0) + 1
            if b.label == "roi":
                val = b.attributes.get("blank")
                if val is not None:
                    roi_blanks.append(str(val))
        summaries.append({
            "image_id": img.id,
            "name": img.name,
            "size": (img.width, img.height),
            "label_counts": label_counts,
            "roi_blank_values": roi_blanks,
        })
    return summaries


def get_boxes_as_dicts(
    root: ET.Element,
    *,
    image_id: Optional[str] = None,
    image_name: Optional[str] = None, 
    labels: Optional[Iterable[str]] = None,
) -> List[Dict[str, object]]:
    """Return selected boxes as plain dictionaries (easy to print/serialize)."""
    out: List[Dict[str, object]] = []
    for b in iter_boxes(root, image_id=image_id, image_name=image_name, labels=labels):
        out.append({
            "image_id": b.image.id if b.image else None,
            "image_name": b.image.name if b.image else None,
            "label": b.label,
            "xtl": b.xtl,
            "ytl": b.ytl,
            "xbr": b.xbr,
            "ybr": b.ybr,
            "z_order": b.z_order,
            "source": b.source,
            "occluded": b.occluded,
            "attributes": dict(b.attributes),
            "width": b.width,
            "height": b.height,
            "area": b.area,
        })
    return out

def get_box_coords(box: Box) -> Tuple[float, float, float, float]:
    """Return (xtl, ytl, xbr, ybr) from a Box dataclass."""
    return box.xtl, box.ytl, box.xbr, box.ybr 


def get_box_center(box: Box) -> Tuple[float, float]:
    """Return center (cx, cy) of the box."""
    return (box.xtl + box.xbr) / 2.0, (box.ytl + box.ybr) / 2.0


def get_box_coords_from_element(box_el: ET.Element) -> Tuple[float, float, float, float]:
    """Parse coordinates directly from a <box> element."""
    def _f(attr: str, default: float = 0.0) -> float:
        v = box_el.get(attr)
        return float(v) if v is not None else default
    return _f("xtl"), _f("ytl"), _f("xbr"), _f("ybr")


def get_boxes_coordinates(
    root: ET.Element,
    *,
    image_id: Optional[str] = None,
    image_name: Optional[str] = None,
    labels: Optional[Iterable[str]] = None,
) -> List[Tuple[Optional[str], Optional[str], str, float, float, float, float]]:
    """Return list of (image_id, image_name, label, xtl, ytl, xbr, ybr) for selected boxes."""
    out: List[Tuple[Optional[str], Optional[str], str, float, float, float, float]] = []
    for b in iter_boxes(root, image_id=image_id, image_name=image_name, labels=labels):
        out.append((b.image.id if b.image else None, b.image.name if b.image else None, b.label, b.xtl, b.ytl, b.xbr, b.ybr))
    return out

# ---------------------------
# Example usage (remove or adapt in your script)
# ---------------------------
if __name__ == "__main__":
    # Demonstration against an XML string
    sample_xml = """
    <annotations>
      <image id="0" name="00002_Q_page_1.tif" width="2477" height="3504">
        <box label="censor" xtl="280.80" ytl="511.70" xbr="2209.40" ybr="842.20" z_order="0" />
        <box label="roi" xtl="108.76" ytl="3101.19" xbr="258.15" ybr="3264.17" z_order="0">
          <attribute name="blank">false</attribute>
        </box>
      </image>
    </annotations>
    """

    tree, root = loads_xml(sample_xml)

    print("Images:")
    for img in iter_images(root):
        print(" -", img)

    print("\nROI boxes in image id=0:")
    for b in iter_boxes(root, image_id="0", labels=["roi"]):
        print(b, b.attributes)

    print("\nAdding attribute reviewer=alice to all ROI boxes...")
    n = add_attribute_to_boxes(root, attr_name="reviewer", attr_value="alice", labels=["roi"])
    print("Updated", n, "boxes")

    print("\nRe-serialised XML:")
    import io
    buf = io.BytesIO()
    save_xml(tree, buf)  # type: ignore[arg-type]
    print(buf.getvalue().decode("utf-8"))
