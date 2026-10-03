"""Provider-free checks of the language-retrieval experiment's mechanisms (ADR-0094): registry, planning, shaping."""

from __future__ import annotations

import json
import unittest

from eval import language

SPEC = language.SearchSpec.of(
    {"text": "query", "relaxable": ["query", "tag"], "fields": ["name", "note"], "collection": "items", "id": "id"}
)


def _item(key: str, name: str, note: object = "") -> dict[str, object]:
    return {"id": key, "name": name, "note": note}


class SpecTests(unittest.TestCase):
    def test_a_documented_spec_without_text_or_relaxable_properties(self):
        spec = language.SearchSpec.of({"text": None, "fields": ["name"], "collection": "items", "id": "id"})
        self.assertIsNone(spec.text)
        self.assertEqual(spec.relaxable, ())
        self.assertIsNone(language.search_text({"query": "x"}, spec))

    def test_items_text_and_identity(self):
        self.assertEqual(language.items({"items": [_item("a", "Taza"), "junk"]}, SPEC), [_item("a", "Taza")])
        self.assertEqual(language.items({"items": "not a list"}, SPEC), [])
        self.assertEqual(language.items(["not a mapping"], SPEC), [])
        self.assertEqual(language.item_text(_item("a", "Taza", 3), SPEC), "Taza")
        self.assertEqual(language.item_text(_item("a", "Taza", "azul"), SPEC), "Taza | azul")
        self.assertEqual(language.item_key(_item("a", "Taza"), SPEC), "id:a")
        self.assertEqual(language.item_key({"id": 7}, SPEC), "id:7")
        for malformed in ({"name": "Taza"}, {"id": True}, {"id": " "}, {"id": 1.5}, {"id": ["a"]}):
            self.assertIsNone(language.item_key(malformed, SPEC))

    def test_search_text_and_relaxed_arguments(self):
        self.assertEqual(language.search_text({"query": "taza"}, SPEC), "taza")
        self.assertIsNone(language.search_text({"query": "  "}, SPEC))
        self.assertIsNone(language.search_text({"query": 3}, SPEC))
        self.assertEqual(language.without_relaxable({"query": "x", "tag": "t", "shop": "s"}, SPEC), {"shop": "s"})
        self.assertIsNone(language.without_relaxable({"shop": "s"}, SPEC))


class RegistryTests(unittest.TestCase):
    def test_one_tag_per_identity_and_the_admission_rule(self):
        registry = language.Registry()
        self.assertEqual(registry.observe("shop", "list", ["a", "b", "c"], ["es", "es", "en"]), 3)
        self.assertEqual(registry.observe("shop", "list", ["a", "d"], ["es", "und"]), 1)
        # en has one record only, below the two-tag minimum; und never counts.
        self.assertEqual(registry.plan("shop", "list"), (["es"], "collection"))
        registry.observe("shop", "list", ["e"], ["en"])
        self.assertEqual(registry.plan("shop", "list"), (["en", "es"], "collection"))
        self.assertEqual(registry.summary(), {"shop.list": {"en": 2, "es": 2, "und": 1}})

    def test_a_share_below_the_minimum_and_the_language_cap(self):
        registry = language.Registry()
        codes = ["es"] * 30 + ["pt"] * 2 + ["en"] * 5 + ["fr"] * 5 + ["de"] * 6
        registry.observe("shop", "list", [str(i) for i in range(len(codes))], codes)
        # pt is 2 of 48, under 10%; four languages qualify and the three largest are kept, ties by code.
        self.assertEqual(registry.plan("shop", "list"), (["es", "de", "en"], "collection"))

    def test_the_assistant_prior_and_no_plan(self):
        registry = language.Registry()
        registry.observe("shop", "orders", ["a", "b"], ["ja", "ja"])
        registry.observe("other", "list", ["a", "b"], ["fr", "fr"])
        copy = registry.copy()
        copy.observe("shop", "orders", ["c"], ["en"])
        self.assertEqual(registry.plan("shop", "list"), (["ja"], "assistant"))
        self.assertEqual(registry.plan("absent", "list"), ([], "none"))
        self.assertEqual(len(registry.tags[("shop", "orders")]), 2)


class HintAndVariantTests(unittest.TestCase):
    def test_a_hint_names_each_language_or_leaves_the_summary(self):
        self.assertEqual(language.hint("List.", [], "none"), "List.")
        self.assertEqual(
            language.hint("List.", ["es", "xx"], "collection"),
            "List. Team has observed that this Action's stored records are written in: Spanish (es), xx (xx).",
        )
        self.assertEqual(
            language.hint("List.", ["ja"], "assistant"),
            "List. Team has observed records written in Japanese (ja) in other reads of this Assistant.",
        )

    def test_the_helper_input_carries_no_record(self):
        body = json.loads(language.variant_input("Find mugs", "Shop.", "list", "List items.", "query", "mug", ["es"]))
        self.assertEqual(body["search_text"], "mug")
        self.assertEqual(body["target_languages"], ["es"])
        self.assertEqual(body["action"], {"id": "list", "summary": "List items.", "search_field": "query"})

    def test_variants_round_robin_and_every_refusal(self):
        answer = {
            "variants": [
                {"lang": "es", "terms": ["taza", "tazas", "ignored third"]},
                {"lang": "en", "terms": ["MUG", "x" * 61]},
                {"lang": "fr", "terms": ["tasse"]},
                {"lang": "pt", "terms": "not a list"},
                "junk",
            ]
        }
        self.assertEqual(language.variants(answer, "mug", ["es", "en", "pt"]), [("es", "taza"), ("es", "tazas")])

    def test_variants_keep_literals_dedupe_and_bound(self):
        answer = {
            "variants": [
                {"lang": "es", "terms": ["taza 500", "taza"]},
                {"lang": "pt", "terms": ["caneca 500", "Taza 500"]},
                {"lang": "en", "terms": ["mug 500", "cup 500"]},
                {"lang": "de", "terms": ["Becher 500", 7]},
            ]
        }
        chosen = language.variants(answer, 'mug 500 "XL"', ["es", "pt", "en", "de"])
        self.assertEqual(chosen, [])
        chosen = language.variants(answer, "mug 500", ["es", "pt", "en", "de"])
        self.assertEqual(chosen, [("es", "taza 500"), ("pt", "caneca 500"), ("de", "Becher 500"), ("en", "cup 500")])
        self.assertEqual(language.variants(None, "mug", ["es"]), [])
        self.assertEqual(language.variants({"variants": "x"}, "mug", ["es"]), [])

    def test_whole_numbers_and_every_quotation_form_are_kept(self):
        def kept(original: str, term: str) -> bool:
            return language.variants({"variants": [{"lang": "es", "terms": [term]}]}, original, ["es"]) != []

        self.assertFalse(kept("mug 500", "taza 1500"))
        self.assertFalse(kept("mug 500", "taza 50"))
        self.assertTrue(kept("mug 500", "taza de 500 ml"))
        self.assertFalse(kept("delta -5", "delta 5"))
        self.assertTrue(kept("delta -5", "delta -5 grados"))
        self.assertFalse(kept("size 1.5", "talla 1,5"))
        for quoted in (
            '"Blue Sky"',
            "'Blue Sky'",
            "“Blue Sky”",
            "‘Blue Sky’",
            "«Blue Sky»",
            "「Blue Sky」",
            "『Blue Sky』",
        ):
            self.assertFalse(kept(f"song {quoted}", "canción Cielo Azul"))
            self.assertTrue(kept(f"song {quoted}", "canción Blue Sky"))

    def test_parse(self):
        self.assertEqual(language.parse('{"a": 1}'), {"a": 1})
        self.assertIsNone(language.parse("{"))
        self.assertIsNone(language.parse(None))


class ShapingTests(unittest.TestCase):
    def test_expansion_keeps_the_original_and_returns_only_new_items_apart(self):
        original = {"items": [_item("a", "Taza azul")]}
        searches = [
            ("es", "taza", {"items": [_item("a", "Taza azul"), _item("b", "Taza roja")]}),
            ("en", "mug", {"items": [_item("b", "Taza roja"), _item("c", "Blue mug")]}),
            ("pt", "caneca", {"error": "none"}),
        ]
        shaped, count = language.expanded(original, SPEC, searches)
        self.assertEqual(count, 2)
        self.assertIs(shaped["assistant_result"], original)
        envelope = shaped["team_search"]
        self.assertEqual([c["id"] for c in envelope["candidates"]], ["b", "c"])
        self.assertEqual(envelope["searched"][2], {"lang": "pt", "text": "caneca", "found": 0})
        self.assertFalse(envelope["truncated"])

    def test_records_without_a_valid_identity_are_never_candidates(self):
        found = {"items": [{"name": "Taza sin id"}, {"id": False, "name": "Taza"}]}
        self.assertEqual(language.expanded({"items": []}, SPEC, [("es", "taza", found)]), ({"items": []}, 0))

    def test_expansion_without_anything_new_returns_the_original_itself(self):
        original = {"items": [_item("a", "Taza")]}
        self.assertEqual(language.expanded(original, SPEC, [("es", "taza", original)]), (original, 0))

    def test_expansion_truncates_its_candidates(self):
        many = {"items": [_item(str(i), f"Taza {i}") for i in range(25)]}
        shaped, count = language.expanded({"items": []}, SPEC, [("es", "taza", many)])
        self.assertEqual((count, shaped["team_search"]["total"], shaped["team_search"]["truncated"]), (20, 25, True))

    def test_ranking_returns_the_best_above_the_threshold(self):
        relaxed = {"items": [_item(str(i), f"Item {i}") for i in range(7)]}
        scores = [0.1, 0.9, 0.5, 0.9, 0.2, 0.8, 0.85]
        shaped, count = language.ranked({"items": []}, SPEC, relaxed, scores, 0.5)
        self.assertEqual(count, 5)
        self.assertEqual([c["item"]["id"] for c in shaped["team_search"]["candidates"]], ["1", "3", "6", "5", "2"])
        self.assertEqual(shaped["team_search"]["candidates"][0]["score"], 0.9)
        self.assertFalse(shaped["team_search"]["truncated"])
        self.assertEqual(language.ranked({"items": []}, SPEC, relaxed, scores, 0.95), ({"items": []}, 0))

    def test_ranking_skips_invalid_and_repeated_identities_before_choosing(self):
        relaxed = {"items": [{"name": "no id"}, _item("a", "A"), _item("a", "A again"), _item("b", "B")]}
        shaped, count = language.ranked({"items": []}, SPEC, relaxed, [0.99, 0.9, 0.95, 0.8], 0.5)
        self.assertEqual(count, 2)
        self.assertEqual(
            [(c["item"]["name"], c["score"]) for c in shaped["team_search"]["candidates"]], [("A", 0.9), ("B", 0.8)]
        )

    def test_ranking_bounds_what_it_ranks(self):
        relaxed = {"items": [_item(str(i), "x") for i in range(language.MAX_RANKED + 1)]}
        shaped, _count = language.ranked({"items": []}, SPEC, relaxed, [0.5] * language.MAX_RANKED, 0.0)
        self.assertEqual(shaped["team_search"]["ranked"], language.MAX_RANKED)
        self.assertTrue(shaped["team_search"]["truncated"])


if __name__ == "__main__":
    unittest.main()
