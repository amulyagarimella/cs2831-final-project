# Corner Stripe Detector

Detecting genomic stripes using corners.

## To run
Unfortunately, the required `.mcool` genomic files are too large to include one here. However, if you obtain one, here are the steps to running the algorithm:
1. **Find known stripe regions/priors.** Run `python3 stripe_prior_detection.20241119.py --help` to see a detailed description of inputs and outputs.
2. **Run stripe detection.** Run `python3 cli.py compute --help` for more details. The output file ending in `.bed` from Step (1) should be used as the `--bed` input for the stripe detector.
