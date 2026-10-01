from resolver.normalize import clean, name_key, house_no


def test_clean_transliterates():
    assert clean("Silver Fóundation Pvt. Ltd.") == "silver foundation pvt ltd"
    assert clean("Tucker [Metro-Transalta]") == "tucker metro transalta"
    assert clean("सिल्वर").isascii()


def test_name_key_ignores_order_case_legal_forms_and_stopwords():
    assert name_key("Yazzie-Empire Enterprises LLC") == name_key("ENTERPRISES  YAZZIE EMPIRE")
    assert name_key("Surgical Care Associates of Topeka Inc") == name_key(
        "surgical care associates of topeka (Inc)")
    assert name_key("Dukes Chadwick Có") == name_key("Dukes Chadwick Co") == "chadwick dukes"
    assert name_key("Meerut Jewellery Private Limited") == name_key("MEERUT JEWELLERY")


def test_name_key_empty_when_only_legal_form():
    assert name_key("LLC") == ""
    assert name_key("") == ""


def test_house_no_variants():
    assert house_no("24800 EUCLID AVE, EUCLID, OH") == "24800"
    assert house_no("02A/3, DELHI, MURADNAGAR") == "2"
    assert house_no("3906a 33rd Ter, Topeka, Kansas") == "3906"
    assert house_no("OH, EUCLID AVE, EUCLID") == ""
    assert house_no("") == ""


from resolver.normalize import name_tokens, nums, addr_parts, is_nonlatin


def test_name_tokens_strips_domains_tags_legal_and_keeps_order():
    assert name_tokens("tuckermetrotransalta.com") == ["tuckermetrotransalta"]
    assert name_tokens(">> www.shreethreads.com") == ["shreethreads"]
    assert name_tokens("surgical care associates of topeka inc #35740") == [
        "surgical", "care", "associates", "topeka"]
    assert name_tokens("LIMITED SUNBEAM EAE") == ["sunbeam", "eae"]


def test_nums_all_numbers_unique_in_order_leading_zeros_stripped():
    assert nums("Door No 276 H 15South Extn Part I") == ["276", "15"]
    assert nums("09-9-232/1/A/1, Ramnagar") == ["9", "232", "1"]
    assert nums("") == []


def test_addr_parts_splits_on_commas_and_cleans():
    assert addr_parts("OH, EUCLID AVE, EUCLID") == ["oh", "euclid ave", "euclid"]
    assert addr_parts(" , N/A") == ["n a"]


def test_is_nonlatin():
    assert is_nonlatin("सिल्वर फाउंडेशन")
    assert not is_nonlatin("Silver Fóundation Lìmited")
