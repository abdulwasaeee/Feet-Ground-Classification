def classify_foot(heel_contact, toe_contact, visible=True):
    if not visible:
        return "Unknown"

    if heel_contact and toe_contact:
        return "Fully Grounded"
    elif heel_contact or toe_contact:
        return "Partially Grounded"
    else:
        return "Not Grounded"


def person_level_classification(left, right):
    if left == "Unknown" and right == "Unknown":
        return "Unknown"

    if left == "Fully Grounded" and right == "Fully Grounded":
        return "Fully Grounded"

    if left == "Not Grounded" and right == "Not Grounded":
        return "Not Grounded"

    return "Partially Grounded"
