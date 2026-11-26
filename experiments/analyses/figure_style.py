"""What the three figure scripts share: tick format and saving."""
from matplotlib.ticker import FuncFormatter


def decimal_axis(axis, fmt='{:g}'):
    axis.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))


def save(fig, output):
    """600 dpi only matters for raster outputs such as .png."""
    fig.savefig(output, dpi=600)
    print(f'-> {output}')
