import matplotlib

matplotlib.use("pgf")
from matplotlib import rcParams, rc
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

plt.rcParams.update(
    {
        "font.size": 20,
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,
        "pgf.preamble": "\n".join(
            [
                r"\usepackage{url}",  # load additional packages
                r"\usepackage{unicode-math}",  # unicode math setup
            ]
        ),
    }
)
