import numpy as np

from screen.option_research_greeks import all_greeks


def test_higher_greeks_match_finite_differences():
    s, k, t, v = 100.0, 103.0, 30 / 365, 0.24
    g = all_greeks(np.array([s]), np.array([k]), np.array([t]), np.array([v]), np.array([True]))
    hs, hv = 1e-3, 1e-5
    sp = all_greeks([s + hs], [k], [t], [v], [True])
    sm = all_greeks([s - hs], [k], [t], [v], [True])
    vp = all_greeks([s], [k], [t], [v + hv], [True])
    vm = all_greeks([s], [k], [t], [v - hv], [True])
    assert np.isclose(g["speed"][0], (sp["bs_gamma"][0] - sm["bs_gamma"][0]) / (2 * hs), rtol=2e-4)
    assert np.isclose(g["vanna_per_1vol"][0], (vp["bs_delta"][0] - vm["bs_delta"][0]) / (2 * hv), rtol=2e-4)
    assert np.isclose(g["vomma_per_1vol2"][0], (vp["bs_vega_per_1vol"][0] - vm["bs_vega_per_1vol"][0]) / (2 * hv), rtol=2e-4)
    assert np.isclose(g["zomma_per_1vol"][0], (vp["bs_gamma"][0] - vm["bs_gamma"][0]) / (2 * hv), rtol=2e-4)


def test_invalid_inputs_are_nan_and_put_delta_is_negative():
    bad = all_greeks([100], [100], [0], [0.2], [True])
    assert np.isnan(bad["bs_gamma"][0])
    put = all_greeks([100], [100], [30 / 365], [0.2], [False])
    assert put["bs_delta"][0] < 0
    assert put["bs_gamma"][0] > 0
