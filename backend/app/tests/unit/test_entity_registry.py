"""注册表只负责查询名称规范化，不推断来源文档或数字归属。"""

import pytest

from service.core.entities.registry import EntityRegistry

pytestmark = pytest.mark.unit


@pytest.fixture
def registry():
    return EntityRegistry({"version": "test", "entities": [
        {"entity_id": "tencent", "canonical_name": "腾讯控股", "kind": "company",
         "aliases": ["腾讯", "Tencent", "Tencent Holdings Limited"], "stock_codes": ["0700.HK"]},
        {"entity_id": "alibaba", "canonical_name": "阿里巴巴", "kind": "company",
         "aliases": ["Alibaba"], "stock_codes": ["9988.HK"]},
        {"entity_id": "tencent-sub", "canonical_name": "腾讯科技有限公司", "kind": "company",
         "aliases": [], "stock_codes": []},
        {"entity_id": "broker", "canonical_name": "甲证券", "kind": "publisher",
         "aliases": [], "stock_codes": []},
    ]})


def test_reviewed_aliases_codes_and_original_quotes(registry):
    text = "Ｔｅｎｃｅｎｔ Holdings Limited、腾 讯和0700.HK属于同一查询对象。"
    mentions = registry.mentions(text)
    assert registry.resolved_ids(mentions) == ["tencent"]
    assert all(text[mention["start"]:mention["end"]] == mention["quote"] for mention in mentions)
    assert registry.mentions("营业收入为603193百万元，年份2023。") == []
    assert registry.mentions("xTencentish") == []
    assert registry.resolved_ids(registry.mentions("The Tencent revenue increased.")) == ["tencent"]


def test_subsidiary_is_separate_and_unregistered_longer_name_not_merged(registry):
    assert registry.resolved_ids(registry.mentions("腾讯科技有限公司")) == ["tencent-sub"]
    assert registry.mentions("腾讯金融科技有限公司") == []


def test_colliding_alias_keeps_all_candidates():
    registry = EntityRegistry({"version": "1", "entities": [
        {"entity_id": "a", "canonical_name": "甲公司", "kind": "company", "aliases": ["华科"]},
        {"entity_id": "b", "canonical_name": "乙公司", "kind": "company", "aliases": ["华科"]},
    ]})
    assert registry.exact("华科") == ["a", "b"]
    assert registry.mentions("华科的收入")[0]["status"] == "ambiguous"
    assert registry.resolved_ids(registry.mentions("华科的收入")) == []
