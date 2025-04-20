FONT_SIZES = {"small": 14, "medium": 18, "large": 24}

PLOT_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": FONT_SIZES.get("medium"),
    "axes.titlesize": FONT_SIZES.get("large"),
    "axes.labelsize": FONT_SIZES.get("large"),
    "xtick.labelsize": FONT_SIZES.get("large"),
    "ytick.labelsize": FONT_SIZES.get("large"),
    "legend.fontsize": FONT_SIZES.get("medium"),
    "figure.titlesize": FONT_SIZES.get("medium"),
    "text.usetex": True,
}

COLORS = {
    "blue": "#D1F9F1",
    "crest": "#FFE2C8",
    "cherry": "#F2CAD8",
    "purple": "#F2ECF8",
    "green": "#DFF2EA",
    "indigo": "#EBEDFB",
    "heritage": "#85B09A",
    "warm_blue": "#00BDB6",
    "warm_crest": "#FFC392",
}

CATEGORY_COLORS = {
    "CN": COLORS.get("blue"),
    "RC": COLORS.get("crest"),
    "CK": COLORS.get("cherry"),
    "GN": COLORS.get("indigo"),
}
