# Job B blind review (10 chats)

For each chat, decide whether each candidate is an **acceptable next action for the support agent**, using the company
playbook descriptions below each option. One candidate is the recorded gold label and one is the models' most common
answer, in random order. **Don't open `jobB_blind_key.json` until you've finished.**

Mark each: `A`, `B`, `both` or `neither`.

## 1

```
Agent: Sure just give me one second
Customer: okay
Agent: And what details would you like to change?
Customer: shipping details
Agent: Ok. Can I have your full name or account ID please?
Customer: Are you able to change the shipping details now ?
Customer: Order ID: 2474295962
Agent: I need to pull up your account first
Customer: Name: Norman Bouchard
[Agent action] Account has been pulled up for Norman Bouchard.
```

- **A:** `verify-identity`: After the account is pulled up, before changing an order, account or subscription: the customer has given details such as account ID, order ID, zip code, username or email to prove who they are.
- **B:** `shipping-status`: Record whether the item has already shipped (it decides whether an order can still be changed or cancelled).

Your verdict: both

## 2

```
Customer: I'm needing to check on the status of my subscription.
Agent: Let me check on that for you. One moment please.
Agent: Can I have your full name or account iD?
Customer: Crystal Minh
[Agent action] Account has been pulled up for crystal minh.
Customer: My account ID is ID7TBHIJQI
Agent: And what is your question about your subscription?
Customer: I would like to check on the status to make sure I still have it.
Agent: Do you have an order ID?
Customer: 2243746561
```

- **A:** `verify-identity`: After the account is pulled up, before changing an order, account or subscription: the customer has given details such as account ID, order ID, zip code, username or email to prove who they are.
- **B:** `subscription-status`: Check the status of the customer's premium subscription (active, amount due, due date).

Your verdict: both

## 3

```
Customer: ??
Agent: Let me check that out for you. Give me one moment
Customer: Are you aware of the Seersucker suits?
Customer: It is a special style?
Customer: Hello
Agent: Yes it appears to be a specialized type of suit. They look very nice.
Agent: I'm sorry for the delay
Customer: Great
Customer: What is the price/
[Agent action] Searching the FAQ pages ...
```

- **A:** `search-pricing`: The FAQ is open and the question is about store pricing.
- **B:** `search-jacket`: The FAQ is open and the question is about jackets.

Your verdict: A

## 4

```
Customer: My credit card keeps getting rejected.
Agent: Ok, I can help you with that
Agent: Does it tell you Invalid Credit Card when you are checking out?
Customer: No.
Agent: Ok, what
Agent: Ok, what seems to be the problem with your card?
Customer: I don't know. It keeps saying that it's getting rejected.
Agent: I understand. I will help fix this issue right now.
[Agent action] Searching the FAQ pages ...
Agent: Ok can you double check and make sure your card is not expired please?
```

- **A:** `log-out-in`: Website problem (cart, search, card, slowness): the first fix to try is asking the customer to log out and back in.
- **B:** `enter-details`: Type in details the customer provided: an amount to pay, a refund address, or details needed for account recovery.

Your verdict: neither

## 5

```
Customer: Yes, it's WQ29HGVOGD
Agent: thanks!
[Agent action] Identity verification in progess ...
Agent: Ok everything in order
Agent: how much do you want to pay Mr. Joyce
Customer: Just the annual fee
Agent: Ok it would be $ 99
Customer: Sounds good, just use the card on file
Agent: understood, then I will proceed the payment now.
Customer: Great, thanks
```

- **A:** `update-account`: Change details on the account: address, name, phone number, payment method, or the subscription's state.
- **B:** `enter-details`: Type in details the customer provided: an amount to pay, a refund address, or details needed for account recovery.

Your verdict: A

## 6

```
[Agent action] Account has been pulled up for David Williams.
Agent: I also need your username, email and order id please
Agent: so that i can validate the purchase
Customer: yeah it's dwilliams052, dwilliams052@email.com, and 9282231792
[Agent action] Purchase validation in progress ...
Agent: Great! What is your membership level?
Customer: i think it's guest.
[Agent action] Searching the FAQ pages ...
[Agent action] System Action: search policy
Agent: What is the purchase date? Guests are allowed to return items for 30 days
```

- **A:** `membership`: The customer has just said their membership level (bronze, silver, gold, guest), to check what they're entitled to.
- **B:** `select-faq`: The right FAQ section has been searched: pick the answer to give the customer.

Your verdict: both

## 7

```
[Agent action] Account has been pulled up for 8UFPPFPH9U.
Agent: I need to validate the purchase, to do this I'm going to need your username, email address, and order ID please.
Customer: okay
Customer: my email is asandersisgreat@gmail.com
Customer: My username is asandersiscool
Customer: my order id is 5773256933
Agent: Great! One moment please.
Agent: Actually I am going to need your full name as well.
Customer: Albert Sanders
Customer: but you can call me Al :)
```

- **A:** `validate-purchase`: For returns, refunds, purchase disputes and shipping problems: the customer has given their username, email and order ID, and the purchase needs confirming.
- **B:** `verify-identity`: After the account is pulled up, before changing an order, account or subscription: the customer has given details such as account ID, order ID, zip code, username or email to prove who they are.

Your verdict: both

## 8

```
Customer: no
Customer: just this one item
Agent: What item is it?
Customer: a tommy hilfiger shirt
Agent: Can I have your full name or Account ID please?
Customer: Albert Sanders
[Agent action] Account has been pulled up for Albert Sanders.
Agent: Also, I will write up a report to let our purchasing department know so they can do a better job
[Agent action] The purchasing department has been notified.
Customer: ok, but what are you going to do about my order
```

- **A:** `make-purchase`: Place the order on the customer's behalf (checkout keeps failing, or an item they want needs ordering).
- **B:** `record-reason`: Record why: the reason for a return, how the customer heard about an issue, or the refund method they chose.

Your verdict: both

## 9

```
[Agent action] Account has been pulled up for Joseph Banter.
[Agent action] Identity verification in progress ...
Agent: Has your order shipped yet?
Customer: The shipping status says delivered.
Agent: Then the new item will have to be sent separately
Agent: Are you a Gold or Silver member?
Customer: silver
Agent: Ok, what would you like to add to the order?
[Agent action] Membership level of Silver has been noted.
Customer: Tommy Hilfiger shirt
```

- **A:** `update-order`: Change an order: upgrade or downgrade shipping, change its address or item, add an item, or cancel it.
- **B:** `make-purchase`: Place the order on the customer's behalf (checkout keeps failing, or an item they want needs ordering).

Your verdict: both

## 10

```
Agent: Thank you Mr. Williams. I also need your account id, order id and membership level please.
[Agent action] Account has been pulled up for David Williams.
Customer: VE17GZRUIV  9262342644  BRONZE
Agent: Thank you. Give me a minute to pull up your info.
Customer: ok
[Agent action] Identity verification in progress ...
Agent: I am sorry Mr. Williams but your account doesn't qualify for an extension at this time.
Customer: I would like to chat with a manager to see if there's anything they can do
Agent: Absolutely. If I can get your phone number I will have a manager call you as soon as possible.
Customer: 490-858-3767
```

- **A:** `enter-details`: Type in details the customer provided: an amount to pay, a refund address, or details needed for account recovery.
- **B:** `notify-team`: Escalate to an internal team (website, purchasing or warehouse) when the agent can't fix it in the chat.

Your verdict: both
