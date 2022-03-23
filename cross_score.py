import pathlib
import multiprocessing as mp

import numpy as np
import cooler
import bioframe

import argparse
import logging


parser = argparse.ArgumentParser(description='Calculate distance-dependent contact marginals of a Hi-C map.')
parser.add_argument('cool', metavar='cool', 
                    type=str, 
                    help='input cooler URI')


parser.add_argument('--chunksize',
                    default=1e6,
                    help='prefix for output files')


parser.add_argument('--prefix', 
                    type=str,
                    default=None,
                    help='prefix for output files')


parser.add_argument('--outfolder', 
                    type=str,
                    default='./',
                    help='output folder')


parser.add_argument('--distbins',
                    default='3e3,3e4,3e5,3e6',
                    help='a comma-separated list of distance bins')


parser.add_argument('--nproc', 
                    type=int,
                    default=None,
                    help='number of processes')



def get_dist_margs(clr_path, lo, hi, dist_bins):
    clr = cooler.Cooler(clr_path)

    bins = clr.bins()[:]
    chunk = clr.pixels()[lo:hi]
    res = clr.binsize

    chunk = cooler.annotate(chunk, bins)
    chunk['balanced'] = np.nan_to_num(chunk['count'] * chunk['weight1'] * chunk['weight2'])
    chunk = chunk[chunk.chrom1 == chunk.chrom2]

    del(clr)

    return _dist_margs(chunk, dist_bins, res)


def _dist_margs(chunk, dist_bins, res): 
    min_bin_id = chunk['bin1_id'].values[0]

    dists = (chunk['bin2_id'] - chunk['bin1_id']) * res
    dist_bin_id = np.searchsorted(dist_bins, dists, 'right') 
    n_dist_bins = len(dist_bins)
    margs_down_loc = np.bincount(
        (chunk['bin1_id'].values - min_bin_id) * n_dist_bins + dist_bin_id,
        weights=np.nan_to_num(chunk['balanced'].values)
        )

    margs_up_loc = np.bincount(
        (chunk['bin2_id'].values - min_bin_id) * n_dist_bins + dist_bin_id,
        weights=np.nan_to_num(chunk['balanced'].values)
            )

    return min_bin_id, margs_down_loc, margs_up_loc


def drop_resolution(clrname):
    name_parts = clrname.split('.')
    if name_parts[-1] in ['cool', 'mcool']:
        name_parts = name_parts[:-1]
    if name_parts[-1].isnumeric():
        name_parts = name_parts[:-1]
    return '.'.join(name_parts)

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

chunksize = int(float(args.chunksize))
clr = cooler.Cooler(args.cool)
bins = clr.bins()[:]
n_pixels = clr.pixels().shape[0]
dist_bins = np.array([int(float(i)) for i in args.distbins.split(',')])

n_dist_bins = len(dist_bins)

chunk_spans = np.r_[np.arange(0, n_pixels, chunksize), n_pixels]

nproc = mp.cpu_count() if args.nproc is None else args.nproc

logging.info(f'Calculating marginals for {args.cool} using {nproc} processes')
with mp.Pool(nproc) as pool:
    out = pool.starmap(get_dist_margs,
        [
            (args.cool, lo, hi, dist_bins)
            for lo, hi in zip(chunk_spans[:-1], chunk_spans[1:])
        ]
    )

margs_up = np.zeros(len(bins) * n_dist_bins)
margs_down = np.zeros(len(bins) * n_dist_bins)

for min_bin_id, margs_down_loc, margs_up_loc in out:
    margs_down[
        min_bin_id * n_dist_bins:
        min_bin_id * n_dist_bins + len(margs_down_loc)
    ] += margs_down_loc

    margs_up[
        min_bin_id * n_dist_bins:
        min_bin_id * n_dist_bins + len(margs_up_loc)
    ] += margs_up_loc


margs_down = margs_down.reshape((len(bins), n_dist_bins)).T
margs_up = margs_up.reshape((len(bins), n_dist_bins)).T


out_folder = pathlib.Path(args.outfolder)
clr_name = pathlib.Path(args.cool.split(':')[0]).name
clr_name = drop_resolution(clr_name)

prefix = clr_name if args.prefix is None else args.prefix
res = clr.binsize

for dist_bin_id in range(n_dist_bins):
    lo = np.r_[0, dist_bins][dist_bin_id]
    hi = np.r_[0, dist_bins][dist_bin_id+1]

    for dir_str, margs in [('up', margs_up), ('down', margs_down), ('both', margs_up+margs_down)]:
        file_name = f'{prefix}.{res}.cross.{dir_str}.{lo}-{hi}.bw'
        logging.info(f'Write output into {file_name}')
        out_df = bins[['chrom', 'start', 'end']].copy()
        out_df['marg'] = margs[dist_bin_id]
        out_df['marg'] = out_df['marg'].mask(bins['weight'].isnull(), np.nan)

        bioframe.io.to_bigwig(
            out_df,
            chromsizes=clr.chromsizes,
            outpath=str(out_folder / file_name))

