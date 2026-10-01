# Encord roundtrip live validation

On 2026-10-01, the opt-in live test ran the shipped workflow locally against Encord and S3. It passed: **1 passed, 3 deselected** in 124.12 seconds. The run used a fresh Encord folder, a fresh Collection, and a fresh S3 prefix. Full receipts and the returned video remain in private operator evidence.

| Stage | Result |
| --- | --- |
| Push | Final receipt; 2 media items uploaded. |
| Curate | Final receipt; width filter selected 1 of 2 items; temporary preset deleted. |
| Pull | Final manifest; 1 item pulled from the recorded Collection. |
| Verify | Passed; 1 expected and matched, with no missing, unexpected, or checksum-mismatched items. |
| Video check | The returned MP4 matched the source SHA-256 `24b8956830fa3524b5843de0b9baff578ee678a992a2569baed7020543b4072c` and decoded to 2 frames. |

The selected UUID set matched the pulled UUID set, and the pull manifest named the Collection recorded by curation. This proves the local client path with real Encord and S3 services for an intrinsic metadata filter.

## Quality filters

The same folder was then prepared in Encord to compute quality metrics. Once folder activity showed **Analysis up to date**, these filters ran through curation, Collection pull, and roundtrip verification:

| Filter | Selection | Verification |
| --- | --- | --- |
| `brightness:0:255` | Both files | 2 expected and matched; checksums available and equal. |
| `brightness:0.2:1` | Video only | 1 expected and matched; checksums available and equal. |
| `sharpness:0:1000000` | Both files | 2 expected and matched; checksums available and equal. |
| `file-size:0:1000` | PNG only | 1 expected and matched; checksums available and equal. |

Encord reported sharpness zero for both simple fixtures. The `sharpness:1:1000000` exclusion check selected nothing and wrote a failed curation receipt, as expected. The file-size check found and fixed a preset mismatch: Encord expects that metric in the `item` domain. A focused local test now checks this payload. After simplifying curation polling, we repeated `file-size:0:1000` against Encord and S3. It again selected the PNG and passed roundtrip verification.

Quality metric computation was started in the Encord app for this folder. The curation, pull, and verification steps ran headlessly against real Encord and S3 services. Execution inside a Kubernetes worker pod remains untested.
