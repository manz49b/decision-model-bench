"""Question set v2: the same questions, labels and gold as v1, but with criteria that state ABCD's own definitions.

Written from ABCD's agent guidelines (``guidelines.json``) and Jev's errors on ``dev150`` only, never ``test``.
This is the fair "write better definitions" competitor to LoRA: the same effort a team would put into criteria
before deciding to fine-tune. The main boundaries it spells out:
- subscription add-ons appearing or disappearing, missing credits and profile changes are ``manage_account``,
  not ``subscription_inquiry``;
- changing the address *on a shipment* is ``shipping_issue``; changing it *on the account* is ``manage_account``;
- delivery date, fees, quantities and adding or removing items on an order are ``order_issue``;
- "return or refund" means returning a received item, not billing disputes or cancelling an order.
"""

FLOWS = {
    "account_access": "Can't log in: forgot their username or password, or needs two-factor authentication reset.",
    "manage_account": (
        "Their account rather than an order: a subscription add-on or service was added or removed without them "
        "wanting it, store credits are missing, a question about shipping on their account, or changing details on "
        "file (account address, name, phone number, payment method)."
    ),
    "order_issue": (
        "An existing order: the delivery time or date is wrong or needs changing, an unexpected fee, the payment "
        "method used for that order, wrong quantity, upgrading or downgrading it, adding an item to it, or removing "
        "an item from it or cancelling it."
    ),
    "product_defect": (
        "Returning an item they received (wrong size, wrong colour, stained), or starting, updating or checking the "
        "status of a refund for a returned item."
    ),
    "purchase_dispute": (
        "Complaining about a price (cheaper at a competitor, or the price dropped), an item out of stock, a promo code "
        "that doesn't work or has expired, or being billed for an item they already returned or never bought."
    ),
    "shipping_issue": (
        "A shipment: checking or confirming where it's being sent, changing the shipping address or item on a shipment, "
        "an item that never arrived, or complaining about shipping cost or cancelling a shipment for a refund."
    ),
    "single_item_query": "A general question about one product type (boots, shirt, jeans or a jacket), with no order.",
    "storewide_query": (
        "A general store question with no specific order or account: pricing policy, membership levels and benefits, "
        "how long things take, or store policies."
    ),
    "subscription_inquiry": (
        "Their premium subscription itself: whether it's active, how much is owed or when it's due, paying the bill, "
        "an extension, or disputing the subscription charge."
    ),
    "troubleshoot_site": "The website isn't working: credit card rejected, cart not updating, search broken, or slow.",
}

SUBFLOWS = {
    "account_access.recover_username": "Forgot their username.",
    "account_access.recover_password": "Forgot their password.",
    "account_access.reset_2fa": "Two-factor authentication isn't working and needs resetting.",
    "manage_account.status_service_added": "A subscription add-on or service was added to their account that they didn't want.",
    "manage_account.status_service_removed": "A subscription add-on or service was removed from their account.",
    "manage_account.status_shipping_question": "A question about shipping or an order, asked about their account status.",
    "manage_account.status_credit_missing": "Store credits they expected are missing from their account.",
    "manage_account.manage_change_address": "Change the address on their account (not on one shipment).",
    "manage_account.manage_change_name": "Change the name on their account.",
    "manage_account.manage_change_phone": "Change the phone number on their account.",
    "manage_account.manage_payment_method": "Change the payment method on file for their account.",
    "order_issue.status_mystery_fee": "An unexpected fee on an order.",
    "order_issue.status_delivery_time": "The delivery time or date of an order is wrong or needs changing.",
    "order_issue.status_payment_method": "A question about the payment method used for an order.",
    "order_issue.status_quantity": "The quantity on an order is wrong.",
    "order_issue.manage_upgrade": "Upgrade an order (e.g. faster shipping).",
    "order_issue.manage_downgrade": "Downgrade an order.",
    "order_issue.manage_create": "Add an item to an existing order.",
    "order_issue.manage_cancel": "Remove an item from an order or cancel the order.",
    "product_defect.refund_initiate": "Start a refund for an item.",
    "product_defect.refund_update": "Change an existing refund (e.g. refund method or amount).",
    "product_defect.refund_status": "Check the status of a refund already requested.",
    "product_defect.return_stain": "Return an item because it's stained.",
    "product_defect.return_color": "Return an item because it's the wrong colour.",
    "product_defect.return_size": "Return an item because it's the wrong size.",
    "purchase_dispute.bad_price_competitor": "Found the item cheaper at a competitor.",
    "purchase_dispute.bad_price_yesterday": "The price dropped after they bought it.",
    "purchase_dispute.out_of_stock_general": "Items in general are out of stock.",
    "purchase_dispute.out_of_stock_one_item": "One specific item they want is out of stock.",
    "purchase_dispute.promo_code_invalid": "A promo code doesn't work.",
    "purchase_dispute.promo_code_out_of_date": "A promo code has expired.",
    "purchase_dispute.mistimed_billing_already_returned": "Billed for an item they already returned.",
    "purchase_dispute.mistimed_billing_never_bought": "Billed for an item they never bought.",
    "shipping_issue.status": "Check or confirm where a shipment is going, or its shipping status.",
    "shipping_issue.manage": "Change the shipping address or item on a shipment.",
    "shipping_issue.missing": "An item never arrived or is missing from the delivery.",
    "shipping_issue.cost": "Complaining about shipping cost, or cancelling a shipment to get a refund.",
    "single_item_query.boots": "A general question about boots.",
    "single_item_query.shirt": "A general question about shirts.",
    "single_item_query.jeans": "A general question about jeans.",
    "single_item_query.jacket": "A general question about jackets.",
    "storewide_query.pricing": "A general question about store pricing.",
    "storewide_query.membership": "A general question about membership levels or benefits.",
    "storewide_query.timing": "A general question about how long things take (shipping, processing).",
    "storewide_query.policy": "A general question about store policies.",
    "subscription_inquiry.status_active": "Is their premium subscription active?",
    "subscription_inquiry.status_due_amount": "How much is owed on their subscription.",
    "subscription_inquiry.status_due_date": "When their subscription payment is due.",
    "subscription_inquiry.manage_pay_bill": "Pay their subscription bill now or renew it.",
    "subscription_inquiry.manage_extension": "Ask for more time to pay their subscription.",
    "subscription_inquiry.manage_dispute_bill": "Dispute a subscription charge.",
    "troubleshoot_site.credit_card": "Their credit card is rejected on the website.",
    "troubleshoot_site.shopping_cart": "The shopping cart isn't updating.",
    "troubleshoot_site.search_results": "Search on the website isn't working.",
    "troubleshoot_site.slow_speed": "The website is slow.",
}


def questions_v2() -> dict:
    from bench.data.abcd import subflow_key, subflows

    expected = {subflow_key(f, s) for f, subs in subflows().items() for s in subs}
    assert set(SUBFLOWS) == expected, set(SUBFLOWS) ^ expected  # same 54 labels as v1, in the same order
    return {
        "flow": {"type": "choice", "instructions": "Which kind of support case is this?", "criteria": FLOWS},
        "subflow": {
            "type": "choice",
            "instructions": "Which specific support case is this? Pick the closest match.",
            "criteria": {k: SUBFLOWS[k] for k in sorted(expected, key=list(SUBFLOWS).index)},
        },
        "is_product_return": {
            "type": "noul",
            "instructions": (
                "Does the customer want to return an item they received (wrong size, wrong colour, stained), or start, "
                "update or check a refund for a returned item? Billing disputes for items already returned or never "
                "bought, cancelling an order, and shipping refunds don't count."
            ),
        },
    }
