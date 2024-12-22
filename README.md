# Corner Stripe Detector

Detecting genomic stripes using corners. 

Stripes are 3D genomic elements that can be detected in pairwise genome "contact maps." 
![3dgenome](https://github.com/user-attachments/assets/22947d16-a2a3-4f29-8677-44c2f92c4062)

However, they are hard to detect at high resolution. I leverage corner detection and known priors to solve this problem efficiently.
![comparison](https://github.com/user-attachments/assets/f38beb53-aca3-4149-8691-0986f83e961e)


## To run
Unfortunately, the required `.mcool` genomic files are too large to include one here. However, if you obtain one, here are the steps to running the algorithm:
1. **Clone repo.**
2. **Install requirements.** `pip install -r requirements.txt`
3. **Find known stripe regions/priors.** Run `python3 stripe_prior_detection.20241119.py --help` to see a detailed description of inputs and outputs.
4. **Run stripe detection.** Run `python3 cli.py compute --help` for more details. The output file ending in `.bed` from Step (1) should be used as the `--bed` input for the stripe detector.
5. **Assess accuracy.** Ensure you have bedtools installed, then run:
`
tail -n +2 [.../result_unfiltered_removeredundant.tsv] | awk -F'\t' 'BEGIN {OFS="\t"} {print $1, int($2), int($3)}' | bedtools sort | bedtools window -a [prior peaks from step (1) .bed] -b stdin -u -w 10000 | wc -l
`. Then, divide by the number of peaks in the prior file to get accuracy percentage. This command takes each of the stripes detected and checks how many of the input known regions are within 10kb (shorter than most genes, but a relatively permissive cutoff) from the stripes.

You can also **score stripes**, though this also requires access to an `.mcool` file. This will assign stripiness values to each stripe and also remove redundant stripes. Run `python3 cli.py score --help` to see a detailed description of inputs and outputs. 

## Visualize results
Check out `example_results/visualize_stripes.ipynb` for an example visualization of the stripes. This requires the `.mcool` file since we need to show the contact map as a background.

To visualize more stripes yourself:
1. **Clone repo.**
2. In the `example_results` folder, open the notebook.
3. Edit the path in cell 337 of the notebook: `lococo_200 = stripeSet.from_results_folder("50_200bp/2024-12-18_11-41-01_stripesearch_resol_200_chr_19_canny_2.0_minL_5_maxW_8_maxpixel_0.95_numcores_4_pvalue_0.1_slow_True_bfilter_1_seed_123456789_windowSize_100_expansionLength_5_method_lococo_", "lococo_200bp")` to another path in `example_results`
4. Change cell 347 to visualize different stripes.

## Acknowledgements
Thank you so much to Sora Yoon. The original code structure and SOTA method is based on [Stripenn](https://github.com/ysora/stripenn).
