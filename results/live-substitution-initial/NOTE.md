This initial run completed live substitution and exact restoration, but reported
PARTIAL because its observation check compared unchanged rows exactly while the
native dispatch and replacement loop used different expert execution kernels
(maximum unaffected-row difference 2.98e-8). Its raw expert-output observations
also used different row orders: token order versus router-slot order. These are
not the final numerical receipt. The final run uses native eager experts for
all conditions and records raw expert outputs in aligned token order.
No adapters were retrained and no scientific thresholds were optimized.
