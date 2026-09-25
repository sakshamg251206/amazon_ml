from src.normalize import clean, name_key, house_no


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
