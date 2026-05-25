-- Scenario 2: nightly analytics rollup.
-- After a marketing push, the 'events' table grew 10× and a NULL surge
-- in events.session_id correlated badly with events.tenant_id, which the
-- planner didn't model. Hash join now spills to disk and a child table
-- regressed from Index Only Scan to Seq Scan.
SELECT t.id, t.name, count(*) AS event_count, sum(e.value_cents) AS revenue
FROM   tenants t
JOIN   events  e ON e.tenant_id = t.id
WHERE  e.created_at >= now() - interval '1 day'
GROUP  BY t.id, t.name
ORDER  BY revenue DESC NULLS LAST
LIMIT  100;
