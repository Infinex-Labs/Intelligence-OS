"""Offline foundation tests — the Phase-1 correctness gate without a camera.

These prove the memory layer's guarantees (§7) on controlled embeddings so the
logic is validated independently of model quality:
  - match-or-mint keeps one person one entity and two people distinct
  - identity survives restart (persistence)
  - merge / split / cascade-delete re-point everything correctly
  - weighted edges grow, decay, and gate candidate->confirmed
Run: .venv/bin/python -m intelligence_os.tests.test_foundation
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from intelligence_os.config import CONFIG
from intelligence_os.identity import FaceDetection, IdentityResolver, l2norm
from intelligence_os.store import Store


def _emb(seed: int, base: np.ndarray | None = None, jitter: float = 0.0) -> np.ndarray:
    """A reproducible 512-d unit vector. `base` + jitter simulates the same person
    across frames; a fresh seed simulates a different person. `jitter` is the L2
    norm of unit-direction noise added to the unit base, so jitter~0.5 gives
    intra-person cosine ~0.8 (realistic for ArcFace), jitter=0 is identical."""
    rng = np.random.default_rng(seed)
    if base is None:
        return l2norm(rng.standard_normal(512).astype(np.float32))
    noise = l2norm(rng.standard_normal(512).astype(np.float32))
    return l2norm(base + jitter * noise)


def _det(emb: np.ndarray) -> FaceDetection:
    return FaceDetection((0, 0, 10, 10), emb, det_score=0.99)


def _fresh_store() -> tuple[Store, str]:
    path = Path(tempfile.mkdtemp()) / "t.db"
    return Store(path), str(path)


def check(name: str, cond: bool):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def test_match_or_mint():
    print("test_match_or_mint")
    store, path = _fresh_store()
    resolver = IdentityResolver(store, threshold=0.45)

    alice_base = _emb(1)
    bob_base = _emb(2)

    # First sighting of Alice -> mint.
    m1 = resolver.resolve(_det(_emb(0, alice_base, jitter=0.5)))
    check("first Alice sighting mints", m1.minted)

    # Same person, new frame (jittered) -> matches the SAME entity.
    m2 = resolver.resolve(_det(_emb(11, alice_base, jitter=0.5)))
    check("second Alice sighting matches (not minted)", not m2.minted)
    check("same entity id for Alice", m1.entity_id == m2.entity_id)

    # A different person -> distinct entity.
    m3 = resolver.resolve(_det(_emb(99, bob_base, jitter=0.5)))
    check("Bob mints a new entity", m3.minted)
    check("Bob != Alice", m3.entity_id != m1.entity_id)

    check("exactly two person entities", len(store.list_entities("person")) == 2)
    store.close()


def test_persistence_across_restart():
    print("test_persistence_across_restart")
    store, path = _fresh_store()
    alice_base = _emb(1)
    r1 = IdentityResolver(store, threshold=0.45)
    aid = r1.resolve(_det(_emb(0, alice_base, jitter=0.5))).entity_id
    store.close()

    # "Next day": brand-new Store object on the same DB file.
    store2 = Store(path)
    r2 = IdentityResolver(store2, threshold=0.45)
    m = r2.resolve(_det(_emb(50, alice_base, jitter=0.5)))
    check("Alice recognized after restart", not m.minted and m.entity_id == aid)
    check("still one entity, not duplicated", len(store2.list_entities("person")) == 1)
    store2.close()


def test_merge_split_delete():
    print("test_merge_split_delete")
    store, _ = _fresh_store()
    a = store.create_entity("person")
    b = store.create_entity("person")
    sa = store.add_signature(a, "face", _emb(1))
    sb = store.add_signature(b, "face", _emb(2))
    o1 = store.add_observation(a, "present", confidence=0.9)
    o2 = store.add_observation(b, "present", confidence=0.9)

    # merge a -> b
    store.merge(a, b)
    check("merged entity marked merged_into", store.get_entity(a)["status"] == f"merged_into:{b}")
    check("signatures re-pointed to b", len(store.entity_signatures(b, "face")) == 2)
    check("observations re-pointed to b", len(store.observations(b)) == 2)
    check("only b is active", [e["entity_id"] for e in store.list_entities("person")] == [b])

    # split: carve sa+o1 back out into a new entity
    new = store.split(b, sig_ids=[sa], observation_ids=[o1])
    check("split created new entity", store.get_entity(new) is not None)
    check("new entity has the carved signature", len(store.entity_signatures(new, "face")) == 1)
    check("new entity has the carved observation", len(store.observations(new)) == 1)
    check("b retains the other observation", len(store.observations(b)) == 1)

    # cascade-delete b
    counts = store.cascade_delete(b)
    check("delete removed b's signatures", counts["signatures"] == 1)
    check("delete removed b's observations", counts["observations"] >= 1)
    check("b marked deleted", store.get_entity(b)["status"] == "deleted")
    store.close()


def test_weight_grow_decay_promote():
    print("test_weight_grow_decay_promote")
    store, _ = _fresh_store()
    p = store.create_entity("person")
    obj = store.create_entity("object")

    cfg = CONFIG.distill
    # Reinforce a 'uses' edge until it confirms.
    n_needed = int(np.ceil(cfg.confirm_weight / cfg.weight_increment))
    rid = None
    for i in range(n_needed):
        rid = store.reinforce_relation("relation", p, "uses", object_entity_id=obj,
                                       obs_confidence=1.0)
    rel = store.find_relation("relation", p, "uses", obj, None)
    check("edge promoted to confirmed", rel["status"] == "confirmed")
    check("weight is capped at 1.0", rel["weight"] <= cfg.weight_cap + 1e-6)

    # Decay far in the future drops it back below confirm.
    future = time.time() + cfg.decay_half_life_days * 86400 * 6
    store.decay_relations(at=future)
    rel2 = store.find_relation("relation", p, "uses", obj, None)
    check("un-reinforced edge decays", rel2["weight"] < rel["weight"])
    check("decayed edge demoted to candidate", rel2["status"] == "candidate")
    store.close()


def test_predicate_legality():
    print("test_predicate_legality")
    store, _ = _fresh_store()
    store.register_predicate("owns", subject_type="person", object_type="object")
    check("person owns object is legal",
          store.is_legal_triple("person", "owns", "object"))
    check("object owns person is illegal",
          not store.is_legal_triple("object", "owns", "person"))
    check("unregistered open predicate allowed",
          store.is_legal_triple("person", "looks_tired", "none"))
    # normalization on the way out (§5b)
    store.register_predicate("napping", canonical="sleeping")
    check("napping canonicalizes to sleeping",
          store.canonical_predicate("napping") == "sleeping")
    store.close()


def test_auto_merge_people():
    print("test_auto_merge_people")
    store, _ = _fresh_store()
    alice_base = _emb(1)
    bob_base = _emb(2)

    # Two fragments of Alice (each close to alice_base) + one Bob.
    a1 = store.create_entity("person")
    store.add_signature(a1, "face", _emb(10, alice_base, jitter=0.4))
    a2 = store.create_entity("person")
    store.add_signature(a2, "face", _emb(20, alice_base, jitter=0.4))
    store.add_observation(a2, "present", confidence=0.9)  # a2 has more obs -> survivor
    store.add_observation(a2, "present", confidence=0.9)
    bob = store.create_entity("person")
    store.add_signature(bob, "face", _emb(30, bob_base, jitter=0.4))

    merged = store.auto_merge_people(threshold=0.45)
    check("merged exactly the two Alice fragments", merged == 1)
    active = [e["entity_id"] for e in store.list_entities("person")]
    check("Alice fragments collapsed to one", a1 not in active or a2 not in active)
    check("Bob stayed separate", bob in active)
    check("two active people remain (Alice + Bob)", len(active) == 2)
    store.close()


def main():
    tests = [test_match_or_mint, test_persistence_across_restart,
             test_merge_split_delete, test_weight_grow_decay_promote,
             test_predicate_legality, test_auto_merge_people]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} foundation tests passed.")


if __name__ == "__main__":
    main()
