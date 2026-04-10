"""My Tag read/write for Rekordbox DB.

Writes energy ratings (and future mood / vocal tags) to Rekordbox's
My Tag system via direct DB access.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


def write_energy_mytags(db: Any, results: list[dict[str, Any]]) -> None:
    """Write energy 1-10 tags to the Rekordbox My Tag system.

    Parameters
    ----------
    db : Rekordbox6Database
        An open pyrekordbox database session.
    results : list[dict]
        Each dict must contain ``energy`` (int 1-10) and ``db_content_id`` (str).
    """
    from pyrekordbox.db6 import tables  # type: ignore[import-untyped]

    now = datetime.now(timezone.utc)
    existing_tags = {t.Name: t for t in db.get_my_tag()}

    # Create Energy parent tag if needed
    if "Energy" not in existing_tags:
        root_tags = [t for t in db.get_my_tag() if t.ParentID == "root"]
        max_seq = max((t.Seq for t in root_tags), default=0)
        energy_parent = tables.DjmdMyTag(
            ID=_new_id(),
            Name="Energy",
            ParentID="root",
            Seq=max_seq + 1,
            Attribute=1,
            UUID=str(uuid.uuid4()),
            rb_data_status=0,
            rb_local_data_status=0,
            rb_local_deleted=0,
            rb_local_synced=0,
            rb_local_usn=100,
            created_at=now,
            updated_at=now,
        )
        db.session.add(energy_parent)
        db.session.flush()
        energy_parent_id = energy_parent.ID
    else:
        energy_parent_id = str(existing_tags["Energy"].ID)

    # Create Energy:N child tags as needed
    energy_tag_ids: dict[int, str] = {}
    needed_values = {r["energy"] for r in results if r.get("energy")}
    for val in needed_values:
        tag_name = f"Energy:{val}"
        if tag_name not in existing_tags:
            child = tables.DjmdMyTag(
                ID=_new_id(),
                Name=tag_name,
                ParentID=energy_parent_id,
                Seq=val,
                Attribute=1,
                UUID=str(uuid.uuid4()),
                rb_data_status=0,
                rb_local_data_status=0,
                rb_local_deleted=0,
                rb_local_synced=0,
                rb_local_usn=100,
                created_at=now,
                updated_at=now,
            )
            db.session.add(child)
            energy_tag_ids[val] = child.ID
        else:
            energy_tag_ids[val] = str(existing_tags[tag_name].ID)

    db.session.flush()

    # Associate tracks with their energy tags
    for r in results:
        energy = r.get("energy")
        if not energy:
            continue
        content_id = str(r["db_content_id"])
        my_tag_id = energy_tag_ids[energy]

        # Remove any old energy associations for this track
        for etid in energy_tag_ids.values():
            db.session.query(tables.DjmdSongMyTag).filter_by(
                ContentID=content_id, MyTagID=etid
            ).delete()

        # Add new association
        db.session.add(
            tables.DjmdSongMyTag(
                ID=_new_id(),
                MyTagID=my_tag_id,
                ContentID=content_id,
                TrackNo=1,
                UUID=str(uuid.uuid4()),
                rb_data_status=0,
                rb_local_data_status=0,
                rb_local_deleted=0,
                rb_local_synced=0,
                rb_local_usn=100,
                created_at=now,
                updated_at=now,
            )
        )


def write_colour(db: Any, content_id: str, colour_id: int) -> None:
    """Set the track colour in Rekordbox."""
    content = db.get_content(ID=content_id)
    if content:
        content.ColorID = colour_id
        content.updated_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_id() -> str:
    """Generate a unique Rekordbox-style ID (string of digits)."""
    return str(uuid.uuid4().int % 4_294_967_295 + 1)
