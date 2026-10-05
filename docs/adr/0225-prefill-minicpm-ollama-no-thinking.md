# ADR-0225: Prefill the MiniCPM Ollama voice alias with closed thinking

Date: 2026-10-05
Status: accepted

## Decision

Refine ADR-0020's desktop voice alias template by appending OpenBMB's explicit
no-thinking assistant prefill, `<think>\n\n</think>\n\n`, after the assistant
role header. Bind those exact bytes in both the committed Modelfile and the
production identity contract. Reject the prior suffixless template, including
an effective Modelfile override, rather than accepting it as voice-ready.

Retain the same MiniCPM Q8 source/blob/alias, stop set, parameters and exact
completion-only capability contract. Retain the separate Gemma main tier and
all input, identity, tool, privacy and performance policies. Rebuild an already
cached source with the existing `tools.setup_minicpm --no-pull` path; this
correction supplies no authority to download or promote a different model.

## Context / why

The owner-authorized Linux trial reached READY and ran the supported Desktop
Responsive identity-off profile, then emitted internal instructions through
speech. Its startup addressing helper also emitted a reasoning block despite
the client passing `think=False`. That flag was already forwarded correctly.
The pinned alias lacked the prompt prefill used by OpenBMB's supported
`enable_thinking=False` path. The existing launcher-owned Go-template setting
selects the alias template but does not manufacture its missing no-thinking
control.

Primary sources: [OpenBMB's chat template](https://huggingface.co/openbmb/MiniCPM5-1B/blob/84cfb63/chat_template.jinja)
and [GGUF mode guidance](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF/blob/075694439cc4b49f0fdf565c7e99e72d8ef29379/README.md).

Public synthetic baseline chat and raw rendering produced reasoning markup;
the explicit closed prefill removed it. The concise arithmetic prompt gave a
short correct reply; the shipped public voice persona gave a short incorrect
reply, 4.2. The addressing question became an exact ACT label. These observations
isolate the prompt-construction defect; they do not prove all internal-prompt
recitation or ambient activation is solved.

## Consequences

- Tests reject the legacy template and render the actual Go template across
  public user/system/history, Unicode and empty-message cases. Focused setup
  gates pass 234 tests with two inherited SWIG warnings; adjacent addressing,
  Ollama cancellation/provider, residency and sanity gates pass 124 tests.
- The same cached alias was rebuilt without pulling. Exact native identity
  passes, as does reasoning-markup absence in synthetic generate and stream.
  Question addressing is exact ACT, but voice-persona arithmetic is incorrect
  in both paths. The initial substring assertion accepted 4.2 as containing 4;
  it was too loose and supplies no arithmetic pass. The daemon was stopped.
- An ambient-statement probe still returns ACT rather than INGEST. Addressing
  quality remains open; never extract an ACT token from hidden reasoning to
  make a malformed classification pass.
- No reasoning-output filter, acoustic/default profile change, microphone
  adapter or enrollment bypass is introduced. Native output-boundary defense
  and a larger disjoint quality evaluation remain separate work.
- The new isolated enrollment is not promoted by this change. Owner live
  validation with that candidate, STOP/talk-over and performance A/B remain
  required; raw recordings, embeddings and native receipts stay off Git.
