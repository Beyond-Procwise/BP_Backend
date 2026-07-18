#!/bin/sh
# Restart the local Node gateway on :3001 against the live bp_sqldb.
#
# Deliberately does NOT export AUTH_BYPASS_SUB: the gateway's .env already holds a real
# cognito_user_id, and an exported value silently wins over dotenv. Setting it to a
# placeholder impersonates a user who does not exist, so every user-scoped route
# (GET /agents/AllAgents and the agent library behind it) returns 404 "User not found".
#
# `pkill` exits non-zero when nothing matched, which turns a successful restart into a
# failed command; the `|| true` keeps that from masking the real result.
set -e
API=/home/muthu/PycharmProjects/beyond-procwaise-Api/beyond_procwaise_api

pkill -f "dist/main.js" 2>/dev/null || true
sleep 2

cd "$API"
[ "$1" = "--build" ] && npm run build

AUTH_BYPASS=true IS_OFFLINE=true NODE_ENV=development \
  nohup node --experimental-global-webcrypto dist/main.js > /tmp/gateway.log 2>&1 &

# Wait for it to actually serve, rather than assuming.
i=0
while [ $i -lt 30 ]; do
  if curl -sf -o /dev/null -H 'x-customer-id: 001' http://localhost:3001/spendiq/metrics; then
    echo "gateway up on :3001"
    grep -iE "NO user" /tmp/gateway.log && echo "WARNING: bypass sub matches no user" || true
    exit 0
  fi
  i=$((i + 1))
  sleep 2
done
echo "gateway did NOT come up - last log lines:"
tail -20 /tmp/gateway.log
exit 1
