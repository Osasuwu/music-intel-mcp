// Runs an actions/github-script body under node with recording stubs, so a test
// exercises the workflow's own JS instead of a Python re-creation of it (port of jarvis#1967).
//
// stdin:  {"script": str, "repo": {"owner", "repo"}, "payload": {...},
//          "issues": ["owner/repo#N", ...]}   — issues.get succeeds only for these
// stdout: {"failed": [str], "notices": [str], "lookups": ["owner/repo#N"]}
"use strict";

const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;

async function main() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  const input = JSON.parse(Buffer.concat(chunks).toString("utf8"));

  const result = { failed: [], notices: [], lookups: [] };
  const existing = new Set(input.issues.map((s) => s.toLowerCase()));
  const core = {
    setFailed: (m) => result.failed.push(String(m)),
    notice: (m) => result.notices.push(String(m)),
    info: () => {},
    warning: () => {},
  };
  const github = {
    rest: {
      issues: {
        get: async ({ owner, repo, issue_number }) => {
          const key = `${owner}/${repo}#${issue_number}`;
          result.lookups.push(key);
          if (!existing.has(key.toLowerCase())) {
            throw Object.assign(new Error("Not Found"), { status: 404 });
          }
          return { data: { number: issue_number } };
        },
      },
    },
  };
  const context = { repo: input.repo, payload: input.payload };

  // The script's own console.log must not corrupt the JSON on stdout.
  console.log = () => {};
  await new AsyncFunction("github", "context", "core", input.script)(github, context, core);
  process.stdout.write(JSON.stringify(result));
}

main().catch((e) => {
  process.stderr.write(String((e && e.stack) || e));
  process.exit(1);
});
