-- Scenario 1: customer order history page.
-- Same query, two different plans.
-- Baseline ran in production for months at sub-100ms; incident plan
-- is what the planner picked after the customers table grew 50× and
-- statistics drifted.
SELECT o.id, o.placed_at, o.total_cents, i.sku, i.qty
FROM   orders        o
JOIN   order_items   i ON i.order_id = o.id
WHERE  o.customer_id = $1
  AND  o.placed_at >= now() - interval '90 days'
ORDER  BY o.placed_at DESC
LIMIT  50;
