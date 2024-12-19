# Corner Stripe Detector

Detecting genomic stripes using corners.

## To run
Unfortunately, the required `.mcool` genomic files are too large to include one here. However, if you obtain one, here are the steps to running the algorithm:
1. **Find known stripe regions/priors.** Run `python3 stripe_prior_detection.20241119.py --help` to see a detailed description of inputs and outputs.
2. **Run stripe detection.** Run `python3 cli.py compute --help` for more details. The output file ending in `.bed` from Step (1) should be used as the `--bed` input for the stripe detector.
3. **Assess accuracy.** Ensure you have bedtools installed, then run:
`
tail -n +2 [.../result_unfiltered_removeredundant.tsv] | awk -F'\t' 'BEGIN {OFS="\t"} {print $1, int($2), int($3)}' | bedtools sort | bedtools window -a [prior peaks from step (1) .bed] -b stdin -u -w 10000 | wc -l
`. Then, divide by the number of peaks in the prior file to get accuracy percentage. This command takes each of the stripes detected and checks how many of the input known regions are within 10kb (shorter than most genes, but a relatively permissive cutoff) from the stripes.
