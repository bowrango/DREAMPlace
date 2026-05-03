##
# @file   LevyPlacer.py
# @brief  Experimental Langevin-style placement engine with clipped Levy noise.
#

from __future__ import annotations

import logging
import math
import time

import numpy as np
import torch

import BasicPlace
import EvalMetrics
import PlaceObj


# Edit these constants while tuning LevyPlacer. The adapter intentionally does
# not expose them as pass-through options, so this file remains the single
# source of truth for the experiment.
#
# Technical motivation:
#
# 1. DREAMPlace as deterministic electron/charge transport.
#    DREAMPlace's density term already turns placement into an electrostatic
#    transport problem: each macro contributes a finite charge cloud, overflow
#    is charge congestion, and the density potential produces the conservative
#    electric field that pushes charge out of crowded regions. The HPWL term
#    adds another conservative force from the net model. Plain gradient descent
#    follows the cold drift
#
#        dx_t = -grad U(x_t) dt,
#
#    where U is the wirelength plus density energy. This is analytically clean,
#    but it can move the empirical macro distribution through narrow transport
#    corridors only by sliding along local gradients. In a rugged placement
#    landscape, that creates lag: the continuous relaxation may look smooth,
#    while the final legalizer still has to perform expensive discrete repairs.
#
# 2. Levy flights as fractional diffusion on placement state space.
#    Replacing Brownian noise with alpha-stable Levy increments corresponds,
#    at the density-evolution level, to a nonlocal fractional generator rather
#    than a local Laplacian diffusion term:
#
#        dX_t = -grad U(X_t) dt + sigma_t dL_t^alpha
#        partial_t rho = div(rho grad U) - sigma_t^alpha (-Delta)^(alpha/2) rho
#
#    up to convention-dependent signs. The nonlocal operator is the useful
#    part: it transports probability mass by rare long jumps, so an ensemble of
#    placements can cross barriers without heating every coordinate as much as
#    Gaussian diffusion would require. The implementation below uses clipped
#    Mantegna-style samples, so it is a controlled engineering approximation
#    of fractional Langevin dynamics rather than an exact sampler.
#
# 3. Link to optimal transport and counterdiabatic steering.
#    Zhong/Frim/DeWeese, arXiv:2512.17087, show that finite-time work
#    extraction in nonconservative overdamped Langevin systems can be written
#    as a Lorentz-force Lagrangian on thermodynamic state space: a kinetic
#    optimal-transport metric term plus a one-form/reversible-work term, with
#    counterdiabatic controls realizing the desired density trajectory.
#    Placement is not an active engine, and this file does not implement their
#    Lorentz force law. The analogy is still valuable: the object we are
#    steering is not one macro, but the evolving distribution of macro charge.
#    The Levy term acts like an external control current that shapes that
#    distribution while the electrostatic gradient supplies the conservative
#    drift. Annealing sigma_t is therefore a protocol design choice: start with
#    enough fractional current to explore alternate transport corridors, then
#    reduce it so the distribution concentrates into a near-legal state before
#    the cheap macro legalizer runs.
#
# 4. Why inverse-sqrt-area weighting.
#    Macro area is a natural proxy for inertia/capacitance in this analogy.
#    Large macros perturb the density field strongly and are expensive to move
#    late, so they should not receive proportionally larger stochastic kicks.
#    Small macros are more fluctuation-sensitive and can often absorb local
#    topological rearrangements. The update multiplies the Levy sample by the
#    x/y node size and then by sqrt(mean_area / area). For square macros this
#    makes absolute kicks roughly area-neutral while relative kicks scale like
#    1 / side_length; for elongated macros it preserves anisotropy by allowing
#    the longer dimension to explore more than the shorter one.
#
# 5. Annealing schedule inspired by finite-time optimal protocols.
#    In the Lorentz/transport view, a good finite-time protocol spreads its
#    transport cost along the path instead of spending the whole budget at the
#    beginning or leaving a large terminal correction. The polynomial schedule
#    below is a first approximation to that idea. LEVY_DECAY_POWER = 1 gives a
#    constant-rate cooldown in iteration time; values < 1 keep exploratory
#    current alive longer; values > 1 quench faster and hand control back to
#    deterministic descent earlier. A future adaptive version could replace
#    iteration time with a measured state-space speed, e.g. overflow decay,
#    density-map displacement, or a Wasserstein-like proxy, to keep transport
#    work closer to constant per step.
#
# 6. Baseline DREAMPlace alignment.
#    The closest match to baseline DREAMPlace is no persistent fractional
#    current:
#
#        LEVY_NOISE_RATIO = 0.0
#        LEVY_MIN_RATIO = 0.0
#
#    That leaves this file as a plain electrostatic gradient-descent variant,
#    so differences are due to optimizer simplification rather than Levy
#    transport. If we want a baseline-like stochastic perturbation while still
#    using this code path, use an almost-Gaussian alpha, uniform node weighting,
#    no residual floor, and a fast decay:
#
#        LEVY_ALPHA = 1.95
#        LEVY_WEIGHT_MODE = "uniform"
#        LEVY_NOISE_RATIO = 0.001 to 0.005
#        LEVY_MIN_RATIO = 0.0
#        LEVY_DECAY_POWER = 2.0
#
#    That setting is intentionally conservative: it resembles DREAMPlace's
#    high-overflow Gaussian kick more than the intended fractional transport
#    experiment. The current defaults below are the more exploratory setting.
LEVY_ALPHA = 1.5
LEVY_NOISE_RATIO = 0.015
LEVY_MIN_RATIO = 0.001
LEVY_DECAY_POWER = 1.0
LEVY_CLIP_RATIO = 1.5
LEVY_WEIGHT_MODE = "inv_sqrt_area"
LEVY_WARMUP = 0
GRAD_CLIP = 0.0


class LevyPlacer(BasicPlace.BasicPlace):
    """A compact DREAMPlace variant driven by preconditioned gradient descent.

    The update is intentionally explicit:

        x_{k+1} = project(x_k - lr_k * grad U(x_k)
                          + sigma_k * S(node) * Levy_alpha)

    where U is DREAMPlace's wirelength + electrostatic density objective and
    S(node) is a configurable size/connectivity weighting.  This keeps the
    electrostatic model intact while adding Langevin-style exploration.
    """

    def __init__(self, params, placedb, timer):
        super(LevyPlacer, self).__init__(params, placedb, timer)
        self.levy_alpha = LEVY_ALPHA
        self.levy_noise_ratio = LEVY_NOISE_RATIO
        self.levy_min_ratio = LEVY_MIN_RATIO
        self.levy_decay_power = LEVY_DECAY_POWER
        self.levy_clip_ratio = LEVY_CLIP_RATIO
        self.levy_weight_mode = LEVY_WEIGHT_MODE
        self.levy_warmup = LEVY_WARMUP
        self.grad_clip = GRAD_CLIP

        if not 0.0 < self.levy_alpha < 2.0:
            raise ValueError("LEVY_ALPHA must be in (0, 2)")

    def __call__(self, params, placedb, learning_rate_value=None):
        all_metrics = []
        iteration = 0

        if params.timing_opt_flag:
            logging.warning("LevyPlacer ignores timing-driven feedback for now")

        if params.global_place_flag:
            for cur_stage, global_place_params in enumerate(params.global_place_stages):
                tt = time.time()
                model = PlaceObj.PlaceObj(
                    0.0,
                    params,
                    placedb,
                    self.data_collections,
                    self.op_collections,
                    global_place_params,
                ).to(self.data_collections.pos[0].device)
                model.train()

                if params.global_place_flag and params.gift_init_flag:
                    init_pos = self.pos[0].view([2, -1])[:, :placedb.num_physical_nodes]
                    init_pos = self.op_collections.gift_init_op.forward(init_pos)
                    self.pos[0][:placedb.num_movable_nodes].data.copy_(
                        init_pos[0, :placedb.num_movable_nodes]
                    )
                    self.pos[0][
                        placedb.num_nodes : placedb.num_nodes + placedb.num_movable_nodes
                    ].data.copy_(init_pos[1, :placedb.num_movable_nodes])

                eval_ops = {
                    "hpwl": self.op_collections.hpwl_op,
                    "overflow": self.op_collections.density_overflow_op,
                }
                if len(placedb.regions) > 0:
                    eval_ops.update(
                        {
                            "density": self.op_collections.fence_region_density_merged_op,
                            "overflow": self.op_collections.fence_region_density_overflow_merged_op,
                            "goverflow": self.op_collections.density_overflow_op,
                        }
                    )

                pos = model.data_collections.pos[0]
                lr = self._initial_learning_rate(model, global_place_params, learning_rate_value)
                weight = self._levy_weight(placedb, model.data_collections)
                best_metric = None
                best_pos = None

                logging.info(
                    "LevyPlacer stage %d: lr=%.4E alpha=%.3f noise=%.4E "
                    "min_noise=%.4E clip=%.3f weight=%s",
                    cur_stage,
                    lr,
                    self.levy_alpha,
                    self.levy_noise_ratio,
                    self.levy_min_ratio,
                    self.levy_clip_ratio,
                    self.levy_weight_mode,
                )

                max_steps = int(model.Lgamma_iteration)
                for local_step in range(max_steps):
                    t0 = time.time()
                    self.op_collections.move_boundary_op(pos)

                    metric = EvalMetrics.EvalMetrics(iteration, (local_step, 0, 0))
                    metric.gamma = model.gamma.data
                    metric.density_weight = model.density_weight.data
                    metric.evaluate(placedb, eval_ops, pos, model.data_collections)
                    model.overflow = metric.overflow.data.clone()

                    if torch.eq(model.density_weight.mean(), 0.0):
                        model.initialize_density_weight(params, placedb)
                        metric.density_weight = model.density_weight.data
                        logging.info("density_weight = %.6E", model.density_weight.mean().item())

                    obj, grad = model.obj_and_grad_fn(pos)
                    metric.objective = obj.data.clone()

                    with torch.no_grad():
                        self._clip_gradient_(grad)
                        pos.add_(grad, alpha=-lr)
                        self._add_levy_noise_(
                            pos=pos,
                            placedb=placedb,
                            data_collections=model.data_collections,
                            weight=weight,
                            local_step=local_step,
                            max_steps=max_steps,
                        )
                        self.op_collections.move_boundary_op(pos)

                    all_metrics.append(metric)
                    if (
                        best_metric is None
                        or float(metric.overflow[-1].item()) < float(best_metric.overflow[-1].item())
                    ):
                        best_metric = metric
                        best_pos = pos.data.clone()

                    if params.plot_flag and (iteration % 100 == 0 or local_step == max_steps - 1):
                        self.plot(params, placedb, iteration, pos.data.clone().cpu().numpy())

                    logging.info(metric)
                    logging.info("LevyPlacer step %.3f ms", (time.time() - t0) * 1000)

                    self._update_gamma(model, placedb, local_step, metric)
                    lr *= float(global_place_params.get("learning_rate_decay", 1.0))
                    iteration += 1

                    if local_step > 20 and float(metric.overflow[-1].item()) < params.stop_overflow:
                        logging.info("LevyPlacer reached stop_overflow at iteration %d", iteration)
                        break

                    if self._diverged(metric, best_metric, params):
                        logging.warning("LevyPlacer divergence guard: restore best position")
                        pos.data.copy_(best_pos)
                        break

                logging.info("LevyPlacer stage %d takes %.3f seconds", cur_stage, time.time() - tt)

        else:
            metric = EvalMetrics.EvalMetrics(iteration)
            metric.evaluate(placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0])
            all_metrics.append(metric)
            logging.info(metric)

        if params.dump_global_place_solution_flag:
            self.dump(params, placedb, self.pos[0].cpu(), "%s.lg.pklz" % (params.design_name()))

        if params.plot_flag:
            self.plot(params, placedb, iteration, self.pos[0].data.clone().cpu().numpy())

        self._legalize(params, placedb, iteration, all_metrics)

        cur_pos = self.pos[0].data.clone().cpu().numpy()
        placedb.apply(
            params,
            cur_pos[0 : placedb.num_movable_nodes],
            cur_pos[placedb.num_nodes : placedb.num_nodes + placedb.num_movable_nodes],
        )
        if params.plot_flag:
            self.plot(params, placedb, iteration + 1, cur_pos)
        return all_metrics

    def _initial_learning_rate(self, model, global_place_params, learning_rate_value):
        base_lr = global_place_params.get("learning_rate", learning_rate_value or 0.01)
        try:
            lr = model.estimate_initial_learning_rate(model.data_collections.pos[0], base_lr)
            lr = float(lr.data.item())
            if not np.isfinite(lr) or lr <= 0:
                raise ValueError("invalid estimated learning rate")
            return lr
        except Exception as err:
            logging.warning("LevyPlacer lr estimate failed (%s); use %.4E", err, base_lr)
            return float(base_lr)

    def _levy_weight(self, placedb, data_collections):
        device = data_collections.pos[0].device
        dtype = data_collections.pos[0].dtype
        n = placedb.num_nodes
        weight = torch.ones(n, dtype=dtype, device=device)

        mode = self.levy_weight_mode.lower()
        movable = slice(0, placedb.num_movable_nodes)
        if mode in ("inv_sqrt_area", "inverse_sqrt_area"):
            area = data_collections.node_areas[movable].clamp(min=1e-12)
            weight[movable] = torch.sqrt(area.mean() / area)
        elif mode == "area":
            area = data_collections.node_areas[movable].clamp(min=1e-12)
            weight[movable] = torch.sqrt(area / area.mean())
        elif mode == "size":
            size = (data_collections.node_size_x[movable] + data_collections.node_size_y[movable]).clamp(min=1e-12)
            weight[movable] = size / size.mean()
        elif mode == "pin":
            pins = data_collections.num_pins_in_nodes[movable].clamp(min=1.0)
            weight[movable] = torch.sqrt(pins / pins.mean())
        elif mode == "uniform":
            pass
        else:
            logging.warning("unknown LEVY_WEIGHT_MODE=%s; use uniform", self.levy_weight_mode)

        weight.clamp_(min=0.25, max=4.0)
        return torch.cat([weight, weight], dim=0)

    def _levy_temperature(self, local_step, max_steps):
        """Protocol temperature for the fractional transport current.

        This deliberately anneals in optimization time rather than waiting for
        legalization to remove overlaps. Early iterations spend exploration
        budget while the density field is high-overflow and many transport
        corridors are still plausible; late iterations preserve only a small
        residual current so deterministic electrostatic descent can settle the
        distribution.
        """
        if local_step < self.levy_warmup:
            return 0.0
        progress = min(max(float(local_step) / max(max_steps - 1, 1), 0.0), 1.0)
        return self.levy_min_ratio + (
            self.levy_noise_ratio - self.levy_min_ratio
        ) * math.pow(1.0 - progress, self.levy_decay_power)

    def _standard_levy(self, shape, device, dtype):
        beta = self.levy_alpha
        sigma_u = (
            math.gamma(1.0 + beta)
            * math.sin(math.pi * beta / 2.0)
            / (
                math.gamma((1.0 + beta) / 2.0)
                * beta
                * math.pow(2.0, (beta - 1.0) / 2.0)
            )
        ) ** (1.0 / beta)
        u = torch.randn(shape, device=device, dtype=dtype) * sigma_u
        v = torch.randn(shape, device=device, dtype=dtype).abs().clamp_(min=1e-12)
        return u / torch.pow(v, 1.0 / beta)

    def _add_levy_noise_(self, pos, placedb, data_collections, weight, local_step, max_steps):
        temp = self._levy_temperature(local_step, max_steps)
        if temp <= 0.0:
            return

        node_size = torch.cat(
            [data_collections.node_size_x, data_collections.node_size_y], dim=0
        ).to(device=pos.device, dtype=pos.dtype)
        node_size = node_size.clamp(min=max(float(placedb.site_width), 1e-6))

        noise = self._standard_levy(pos.size(), pos.device, pos.dtype)
        noise.mul_(node_size).mul_(weight).mul_(temp)
        clip = node_size * self.levy_clip_ratio
        noise.clamp_(min=-clip, max=clip)

        self._zero_nonmovable_(noise, placedb)
        pos.add_(noise)

    def _zero_nonmovable_(self, update, placedb):
        update[placedb.num_movable_nodes : placedb.num_nodes - placedb.num_filler_nodes].zero_()
        update[
            placedb.num_nodes + placedb.num_movable_nodes : 2 * placedb.num_nodes - placedb.num_filler_nodes
        ].zero_()

    def _clip_gradient_(self, grad):
        if self.grad_clip <= 0.0:
            return
        norm = grad.norm(p=2)
        if torch.isfinite(norm) and norm > self.grad_clip:
            grad.mul_(self.grad_clip / norm)

    def _update_gamma(self, model, placedb, local_step, metric):
        if len(placedb.regions) > 0 and getattr(metric, "goverflow", None) is not None:
            model.op_collections.update_gamma_op(local_step, metric.goverflow)
        elif getattr(metric, "overflow", None) is not None:
            model.op_collections.update_gamma_op(local_step, metric.overflow)
        else:
            model.op_collections.precondition_op.set_overflow(metric.overflow)

    def _diverged(self, metric, best_metric, params):
        if best_metric is None:
            return False
        overflow = float(metric.overflow[-1].item())
        best_overflow = float(best_metric.overflow[-1].item())
        hpwl = float(metric.hpwl.item())
        best_hpwl = float(best_metric.hpwl.item())
        if overflow < max(params.stop_overflow, best_overflow):
            return False
        return hpwl > best_hpwl * 2.0 and overflow > best_overflow * 1.2

    def _legalize(self, params, placedb, iteration, all_metrics):
        if not params.legalize_flag:
            return

        tt = time.time()
        self.pos[0].data.copy_(self.op_collections.legalize_op(self.pos[0]))
        legal = self.op_collections.legality_check_op(self.pos[0])
        logging.info("legalization takes %.3f seconds", time.time() - tt)
        if not legal:
            raise RuntimeError("LevyPlacer legalization failed legality check")

        metric = EvalMetrics.EvalMetrics(iteration)
        metric.evaluate(placedb, {"hpwl": self.op_collections.hpwl_op}, self.pos[0])
        all_metrics.append(metric)
        logging.info(metric)


# Drop-in alias for Placer.py, which imports NonLinearPlace.NonLinearPlace.
NonLinearPlace = LevyPlacer
