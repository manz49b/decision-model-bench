"""Job B, question set v2: the 30 actions described by *when* the company's playbook uses them.

Written from ABCD's agent guidelines (``guidelines.json``), which are procedural: each action has a place in a
flow. The criteria spell out triggers and order (pull up the account first, verify before changing anything,
search the FAQ section before selecting an answer).

One edit pass after checking Jev's errors on dev150 (0.613), then frozen: Jev kept re-picking steps the chat already
shows as done (``verify-identity`` again, ``search-shirt`` when the next step is ``select-faq``), and mixed up the
website-fix order. The instructions now say done steps are never repeated, and the fix order is explicit.
"""

INSTRUCTIONS = (
    "Which action should the support agent take next in this chat? Lines starting [Agent action] are steps the agent "
    "has already done: never repeat one of them, pick the step that comes after it in the playbook."
)

V2 = {
    "pull-up-account": "First step in almost every chat: the customer has just given their full name or account ID and their account isn't pulled up yet.",
    "verify-identity": "After the account is pulled up, before changing an order, account or subscription: the customer has given details such as account ID, order ID, zip code, username or email to prove who they are.",
    "validate-purchase": "For returns, refunds, purchase disputes and shipping problems: the customer has given their username, email and order ID, and the purchase needs confirming.",
    "ask-the-oracle": "Check in the system whether the customer's claim is a valid error (an unexpected fee, a delivery date, a service added or removed, missing credit), after their identity is verified.",
    "membership": "The customer has just said their membership level (bronze, silver, gold, guest), to check what they're entitled to.",
    "search-faq": "The customer asked a general question answered in the FAQ (about a product or the store), with no account needed: open the FAQ.",
    "search-jeans": "The FAQ is open and the question is about jeans.",
    "search-shirt": "The FAQ is open and the question is about shirts.",
    "search-boots": "The FAQ is open and the question is about boots.",
    "search-jacket": "The FAQ is open and the question is about jackets.",
    "search-pricing": "The FAQ is open and the question is about store pricing.",
    "search-timing": "The FAQ is open and the question is about how long things take.",
    "search-membership": "The FAQ is open and the question is about membership levels or benefits.",
    "search-policy": "The FAQ is open and the question is about store policy.",
    "select-faq": "The right FAQ section has been searched: pick the answer to give the customer.",
    "log-out-in": "Website problem (cart, search, card, slowness): the first fix to try is asking the customer to log out and back in.",
    "try-again": "Website problem, and the customer has already logged out and back in: ask them to try the action again.",
    "instructions": "Website problem, only after logging out and in and trying again have both failed: give step-by-step instructions (e.g. clearing cookies or cache).",
    "notify-team": "Escalate to an internal team (website, purchasing or warehouse) when the agent can't fix it in the chat.",
    "make-purchase": "Place the order on the customer's behalf (checkout keeps failing, or an item they want needs ordering).",
    "record-reason": "Record why: the reason for a return, how the customer heard about an issue, or the refund method they chose.",
    "enter-details": "Type in details the customer provided: an amount to pay, a refund address, or details needed for account recovery.",
    "offer-refund": "Give the customer a refund of a specific amount (for a wrong charge, a price dispute or a missing credit).",
    "update-order": "Change an order: upgrade or downgrade shipping, change its address or item, add an item, or cancel it.",
    "update-account": "Change details on the account: address, name, phone number, payment method, or the subscription's state.",
    "shipping-status": "Record whether the item has already shipped (it decides whether an order can still be changed or cancelled).",
    "send-link": "Send the customer a link to their account page so they can check this themselves next time, usually near the end.",
    "make-password": "Generate a new password for a customer who forgot theirs, once their identity is confirmed.",
    "promo-code": "Generate a promo code for the customer (compensation, or to replace one that didn't work).",
    "subscription-status": "Check the status of the customer's premium subscription (active, amount due, due date).",
}
