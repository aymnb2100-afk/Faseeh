# Security Policy

## Supported version

Faseeh is a website, not a package with releases: only what is on the `main` branch (the site at
https://faseeh.space and its analysis servers) is supported. Security fixes go to `main` directly.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Report it privately through GitHub: open the **Security** tab of this repository and choose
**Report a vulnerability**. Describe what you found, how to reproduce it, and what it could affect.

What happens next:

- We reply within 7 days to confirm we received it.
- If it is a real problem, we fix it on `main` and tell you when the fix is live.
- If we decide it is not a vulnerability, we explain why.

## What is in scope

- The website (`index.html`, `review.html`) and its handling of what the analysis server returns.
- The analysis server (`server/`): uploads, limits, the team key, and the saving of recordings.
- The GitHub workflow that deploys the server.

Recordings are kept only when the reader agrees, in a private dataset. No account, password or payment
data exists in Faseeh.

## Secrets

No token or key belongs in this repository. Server secrets live in the hosting settings and in a private
settings file on the team's machine that is never uploaded.
