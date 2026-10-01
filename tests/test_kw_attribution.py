"""De-attribution of summary statements (spec §5, attribution-robust faithfulness).

NLI with a first-person review as premise gives near-zero entailment to "Some
reviewers found X", even when a review says X. These cases come from real
Llama-3.2-3B summaries (30 Sep diagnostic) and from the frames the splitter
itself produces ("Two reviewers ..." becomes "Some reviewers ...").
"""

import pytest

from fairsum.kwsum.attribution import deattribute


@pytest.mark.parametrize("text, expected", [
    # communication verb + "that": keep only the reported clause
    ("Some reviewers mention that the product can be prone to falling apart quickly, "
     "especially if not handled carefully",
     "The product can be prone to falling apart quickly, especially if not handled carefully"),
    ("One reviewer notes that it caused headaches", "It caused headaches"),
    ("Several reviewers also reported that they no longer needed their rescue inhaler",
     "I no longer needed my rescue inhaler"),
    ("Reviews suggest that the pump breaks easily", "The pump breaks easily"),
    # any other verb: the reviewers become "I", as in the reviews themselves
    ("Some reviewers found the scent to be unappealing or overpowering",
     "I found the scent to be unappealing or overpowering"),
    ("others loved the fragrance", "I loved the fragrance"),
    ("However, many reviewers praised its ability to make their hair feel soft, silky, and manageable",
     "I praised its ability to make my hair feel soft, silky, and manageable"),
    ("Overall, users highly recommend the product for its convenience and reliability",
     "I highly recommend the product for its convenience and reliability"),
    ("However, some mention minor issues with the design", "I mention minor issues with the design"),
    ("Some users experienced weight gain", "I experienced weight gain"),
    ("Many found it effective", "I found it effective"),
    # verb agreement after the subject changes
    ("Many patients were able to stop using their inhalers", "I was able to stop using my inhalers"),
    ("Another user has experienced hair loss", "I have experienced hair loss"),
    ("One reviewer notes it caused headaches", "I note it caused headaches"),
    ("A reviewer wishes it came in a larger size", "I wish it came in a larger size"),
    ("Some parents are happy with the results", "I am happy with the results"),
    # "according to" and evaluative passives
    ("According to some reviewers, the pump breaks easily", "The pump breaks easily"),
    ("Overall, the product is considered a great value, especially considering its quality",
     "The product is a great value, especially considering its quality"),
    ("It is widely regarded as an effective treatment", "It is an effective treatment"),
])
def test_attribution_frames_are_removed(text, expected):
    assert deattribute(text) == expected


@pytest.mark.parametrize("text", [
    "The Luxury Bath Pouf is a gentle, effective exfoliating scrubber that comes in various sizes",
    "The product received mixed reviews from users",   # a claim about the balance of opinion
    "Most reviews are positive",
    "It caused severe nausea",
    "Some of the pads fell apart",                      # "some" of the product, not of the reviewers
    "Users with sensitive skin reported rashes",        # a frame the rules do not know: left alone
    # a bare singular noun is an adjective here, not a reviewer (seen in 3B output)
    "Overall, individual experiences with Actos vary widely",
    "User reviews praise the scent",
])
def test_statements_without_a_known_frame_are_unchanged(text):
    assert deattribute(text) == text


def test_a_frame_with_nothing_after_it_is_left_alone():
    assert deattribute("Some reviewers agreed") == "Some reviewers agreed"
