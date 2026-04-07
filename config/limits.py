from config.specs import LimitPolicy


CHECK_POINTS = (200, 1000, 4000)
DEFECT_TYPES = ("Margin Out", "Curved Out", "No Signal", "Nan")

CONTROL_LIMIT_SPECS = (
    ("Digital MIC", "UCL", -35, -36, -35, 0.5),
    ("Digital MIC", "LCL", -39, -38, -39, "-"),
    ("Analog MIC", "UCL", -14.5, -9, -8, 1.0),
    ("Analog MIC", "LCL", -18.5, -11, -12, "-"),
)

CPK_LIMITS = {
    "analog": (-11, -9),
    "digital": (-38, -36),
}

NO_SIGNAL_LIMITS = {
    "analog": -30,
    "digital": -45,
}

THD_LIMITS = {
    "analog": 1.0,
    "digital": 0.5,
}

LIMIT_POLICY = LimitPolicy(
    check_points=CHECK_POINTS,
    defect_types=DEFECT_TYPES,
    control_limit_specs=CONTROL_LIMIT_SPECS,
    cpk_limits=CPK_LIMITS,
    no_signal_limits=NO_SIGNAL_LIMITS,
    thd_limits=THD_LIMITS,
)
