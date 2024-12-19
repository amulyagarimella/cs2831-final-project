import typer
from typing import List, Optional
import stripeDetector 
import multiprocessing
from datetime import datetime
from pathlib import Path
import pandas as pd

app = typer.Typer()

def current_timestamp ():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def dir_current_timestamp (dir_name="", basedir_name=None, makedir=True):
    ct = current_timestamp()
    if basedir_name is not None:
        basedir = Path(basedir_name).resolve()
    else:
        basedir = Path.cwd()
    if len(dir_name) > 0:
        dir_name = dir_name + "_"
    outpath = ( basedir / f"{ct}_{dir_name}" ).resolve()
    if makedir:
        outpath.mkdir(parents=True)
    return outpath


@app.command('compute')
def execute(
    cool: str = typer.Option(..., "--cool",help="Path to cool file"),
    out: str = typer.Option(..., "--out", "-o", help="Base path to CREATE output directory"),
    bedfilename: str = typer.Option(..., "--bed", "-b", help="Path to bed file containing candidate stripe coordinates."),
    signalfilename: str = typer.Option("", "--signal", "-g", help="Path to file with signal per bin."),
    region: str = typer.Option("", "--region", "-r", help="Genomic region (e.g., chr1:135010000-136000000)"),
    norm: str = typer.Option('KR',"--norm",help="Normalization method. It should be one of the column name of Cooler.bin(). Check it with Cooler.bins().columns (e.g., KR, VC, VC_SQRT)"),
    chrom: str = typer.Option('all', "--chrom", "-k", help="Set of chromosomes. e.g., 'chr1,chr2,chr3', 'all' will generate stripes from all chromosomes"),
    canny: float = typer.Option(2.0, "--canny", "-c", help="Canny edge detection parameter."),
    minL: int = typer.Option(10,'--minL','-l', help="Minimum length of stripe."),
    maxW: int = typer.Option(8, '--maxW','-w', help="Maximum width of stripe."),
    maxpixel: str = typer.Option('0.95,0.96,0.97,0.98,0.99','--maxpixel','-m', help="Percentiles of the contact frequency data to saturate the image. Separated by comma"),
    numcores: int = typer.Option(multiprocessing.cpu_count(), '--numcores','-n', help='The number of cores will be used.'),
    pvalue: float = typer.Option(0.1,  '--pvalue','-p', help='P-value cutoff for stripe.'),
    mask: str = typer.Option('0', "--mask", help='Column coordinates to be masked. e.g., chr9:12345678-12345789'),
    slow: bool= typer.Option(False, "-s", help='Use if system memory is low.'),
    bfilter: int=typer.Option(3,"--bfilter",'-b',help="Mean filter size. should be an odd number"),
    seed: int=typer.Option(123456789, "--seed", help="Seed used to initialize the PRNG."),
    windowSize: int=typer.Option(400, "--windowSize", help="Window size in pixels for the stripe detection."),
    expansionLength_px: int=typer.Option(5, "--expansionLength", help="Expansion length in px out from priors for the stripe detection. (i.e. length to expand on each side)"),
    method: str = typer.Option('lococo', "--method", help="Method to detect stripe. It should be one of the following: 'lococo', 'lococo_torch', 'harris', 'canny'")
):
    """Finds stripe coordinates from 3D genomic data
    """
    print(chrom)
    #cool_basename_for_outdir = "cool_" + Path(cool).stem.rsplit('.', maxsplit=1)[0]
    #bedfilename_basename_for_outdir = "bed_" + Path(bedfilename).stem.split('.', maxsplit=1)[0]
    #signalfilename_basename_for_outdir = "signal_" + Path(signalfilename).stem.split('.', maxsplit=1)[0]
    if len(region) > 0:
        region_for_outdir = region.replace(':', '_').replace('-', '_') + "_"
    else:
        region_for_outdir = ""
    
    resol_for_outdir = f"resol_{cool.split('/')[-1]}"
    chrom_for_outdir = "chr_" + chrom.replace(',', '_')
    canny_for_outdir = f"canny_{canny}"
    minL_for_outdir = f"minL_{minL}"
    maxW_for_outdir = f"maxW_{maxW}"
    maxpixel_for_outdir = f"maxpixel_{maxpixel}"
    numcores_for_outdir = f"numcores_{numcores}"
    pvalue_for_outdir = f"pvalue_{pvalue}"
    #mask_for_outdir = f"mask_{mask}"
    slow_for_outdir = f"slow_{slow}"
    bfilter_for_outdir = f"bfilter_{bfilter}"
    seed_for_outdir = f"seed_{seed}"
    windowSize_for_outdir = f"windowSize_{windowSize}"
    expansionLength_px_for_outdir = f"expansionLength_{expansionLength_px}"
    method_for_outdir = f"method_{method}"

    outdir_full = str(dir_current_timestamp(dir_name = f"stripesearch_{resol_for_outdir}_{region_for_outdir}{chrom_for_outdir}_{canny_for_outdir}_{minL_for_outdir}_{maxW_for_outdir}_{maxpixel_for_outdir}_{numcores_for_outdir}_{pvalue_for_outdir}_{slow_for_outdir}_{bfilter_for_outdir}_{seed_for_outdir}_{windowSize_for_outdir}_{expansionLength_px_for_outdir}_{method_for_outdir}", 
                                            basedir_name = out))

    stripeDetector.compute(cool, outdir_full, bedfilename, signalfilename, region, norm, chrom, canny, minL, maxW, maxpixel, numcores, pvalue, mask, slow, bfilter, seed, windowSize, expansionLength_px, method)
"""
@app.command('seeimage')
def seeimag(
        cool: str = typer.Option(..., "--cool",help="Path to cool file"),
        position: str = typer.Option(..., "--position",'-p', help="Genomic position (e.g., chr1:135010000-136000000)"),
        maxpixel: str = typer.Option('0.95,0.96,0.97,0.98,0.99',"--maxpixel",'-m', help="Quantile for the pixel saturation. (e.g., 0.95)"),
        out: str = typer.Option('./heatmap.png', "--out", "-o", help="Path to output directory"),
        norm: str = typer.Option('KR',"--norm",help="Normalization method. It should be one of the column name of Cooler.bin(). Check it with Cooler.bins().columns (e.g., KR, VC, VC_SQRT)"),
        slow: bool= typer.Option(False,'-s' , help='Use if system memory is low.')

):
    \""" Draws heatmap image of given position and color saturation parameter (maxpixel).
    \"""
    seeimage.seeimage(cool, position, maxpixel, norm, out, slow)
    return 0"""

@app.command('score')
def scoring(
    result_df_path: str = typer.Option(..., "--result", help="Path to result file"),
    cool: str = typer.Option(..., "--cool",help="Path to cool file"),
    bedfilename: str = typer.Option(..., "--bed", "-b", help="Path to bed file containing candidate stripe coordinates."),
    signalfilename: str = typer.Option("", "--signal", "-g", help="Path to file with signal per bin."),
    region: str = typer.Option("", "--region", "-r", help="Genomic region (e.g., chr1:135010000-136000000)"),
    norm: str = typer.Option('KR',"--norm",help="Normalization method. It should be one of the column name of Cooler.bin(). Check it with Cooler.bins().columns (e.g., KR, VC, VC_SQRT)"),
    chrom: str = typer.Option('all', "--chrom", "-k", help="Set of chromosomes. e.g., 'chr1,chr2,chr3', 'all' will generate stripes from all chromosomes"),
    canny: float = typer.Option(2.0, "--canny", "-c", help="Canny edge detection parameter."),
    minL: int = typer.Option(10,'--minL','-l', help="Minimum length of stripe."),
    maxW: int = typer.Option(8, '--maxW','-w', help="Maximum width of stripe."),
    maxpixel: str = typer.Option('0.95,0.96,0.97,0.98,0.99','--maxpixel','-m', help="Percentiles of the contact frequency data to saturate the image. Separated by comma"),
    numcores: int = typer.Option(multiprocessing.cpu_count(), '--numcores','-n', help='The number of cores will be used.'),
    pvalue: float = typer.Option(0.1,  '--pvalue','-p', help='P-value cutoff for stripe.'),
    mask: str = typer.Option('0', "--mask", help='Column coordinates to be masked. e.g., chr9:12345678-12345789'),
    slow: bool= typer.Option(False, "-s", help='Use if system memory is low.'),
    bfilter: int=typer.Option(3,"--bfilter",'-b',help="Mean filter size. should be an odd number"),
    seed: int=typer.Option(123456789, "--seed", help="Seed used to initialize the PRNG."),
    windowSize: int=typer.Option(400, "--windowSize", help="Window size in pixels for the stripe detection."),
    expansionLength_px: int=typer.Option(5, "--expansionLength", help="Expansion length in px out from priors for the stripe detection. (i.e. length to expand on each side)"),
    method: str = typer.Option('lococo', "--method", help="Method to detect stripe. It should be one of the following: 'lococo', 'lococo_torch', 'harris', 'canny'")
):
    """ Calculates p-value and stripiness of given stripes based on given 3D genome conformation data.
    """
    outdir_full = result_df_path.rsplit("/",1)[0]
    result_df = pd.read_csv(result_df_path, sep="\t")
    #score.getScore(cool, coordinates, norm, numcores, out, mask)
    stripeDetector.score_only(result_df, cool, outdir_full, bedfilename, signalfilename, region, norm, chrom, canny, minL, maxW, maxpixel, numcores, pvalue, mask, slow, bfilter, seed, windowSize, expansionLength_px, method)


def main():
    app()

if __name__ == "__main__":
    app()
