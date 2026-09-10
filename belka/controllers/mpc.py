"""3D MPC controller: 13-мерное состояние, 6DoF динамика, acados SQP."""

import hashlib
import time
from pathlib import Path
from typing import Optional

import numpy as np

from belka.common import (
    RobotParams3D,
    RobotState3D,
    ValveOpenings,
    quat_normalize,
)
from belka.actuation import (
    active_thruster_mask,
    symbolic_applied_thruster_forces,
    symbolic_desired_thruster_forces,
    valve_openings_to_forces,
)
from belka.shared.controllers import ControlHistory, Controller
from belka.shared.observers import WrenchEstimate3D
from belka.shared.planners import TrajectoryPlanner


class MPCController3D(Controller):
    """MPC для 6DoF модели: состояние [p, q, v, omega], управление - тяги движителей."""

    FIXED_SOLVER_OPTIONS = {
        "integrator_type": "DISCRETE",
        "qp_solver": "PARTIAL_CONDENSING_HPIPM",
        "hessian_approx": "GAUSS_NEWTON",
        "nlp_solver_type": "SQP",
        "nlp_solver_max_iter": 10,
        "qp_solver_warm_start": 1,
        "nlp_solver_tol_stat": 1e-6,
        "nlp_solver_tol_eq": 1e-6,
        "nlp_solver_tol_ineq": 1e-6,
        "nlp_solver_tol_comp": 1e-6,
    }

    def __init__(
        self,
        robot_params: RobotParams3D,
        state_weight,
        thruster_weight,
        thruster_delta_weight,
        planner: TrajectoryPlanner,
        terminal_weight=None,
        horizon: int = 12,  # N — длина горизонта MPC
        dt: float = 1 / 20,  # шаг дискретизации MPC
        solver_opts: Optional[dict] = None,
        active_thrusters: tuple[int, ...] | None = None,
        build_solver_if_missing: bool = True,
    ) -> None:
        """Инициализировать пространственный MPC, где u — раскрытия клапанов."""
        self.history = ControlHistory()
        self.planner = planner
        self.last_prediction = None
        self.last_prediction_controls = None
        self.solve_statuses: list[int] = []
        self.solve_times: list[float] = []
        self.robot_params = robot_params
        self.horizon = horizon  # N
        self.dt = dt  # шаг дискретизации

        self.nx = 13  # размерность вектора состояния: p(3)+q(4)+v(3)+ω(3)
        self.nu = robot_params.nu  # размерность управления: количество движителей
        self.acados_nx = self.nx + self.nu  # расширенное состояние [x; u_prev]
        self.acados_ny = self.nx + self.nu + self.nu  # размер yref: [x; u; Δu]
        self.max_opening = 1.0
        self.slew_limit = float(robot_params.valve_response_per_s * dt)
        self.side_force_groups = tuple(
            tuple(int(i) for i in group) for group in robot_params.side_force_groups
        )
        self.active_mask = active_thruster_mask(self.nu, active_thrusters)
        self.build_solver_if_missing = bool(build_solver_if_missing)

        state_weight = self._validate_weight(
            state_weight, (self.nx, self.nx), "state_weight"
        )
        thruster_weight = self._validate_weight(
            thruster_weight, (self.nu, self.nu), "thruster_weight"
        )
        thruster_delta_weight = self._validate_weight(
            thruster_delta_weight, (self.nu, self.nu), "thruster_delta_weight"
        )
        if terminal_weight is None:
            terminal_weight = state_weight.copy()
        else:
            terminal_weight = self._validate_weight(
                terminal_weight, (self.nx, self.nx), "terminal_weight"
            )

        self.state_weight = state_weight
        self.thruster_weight = thruster_weight
        self.thruster_delta_weight = thruster_delta_weight
        self.terminal_weight = terminal_weight

        self.solver_opts = dict(solver_opts or {})
        for option, fixed_value in self.FIXED_SOLVER_OPTIONS.items():
            if option in self.solver_opts and self.solver_opts[option] != fixed_value:
                raise ValueError(
                    f"{option} is fixed at {fixed_value!r} for reproducible MPC"
                )
        self.effective_solver_options = {
            **self.FIXED_SOLVER_OPTIONS,
            "print_level": 0,
            **self.solver_opts,
        }
        self._setup_solver(
            self.state_weight,
            self.thruster_weight,
            self.thruster_delta_weight,
            self.terminal_weight,
        )

    @staticmethod
    def _validate_weight(weight, shape, name):
        """Проверить, что матрица весов квадратная, симметричная и PSD."""
        matrix = np.asarray(weight, dtype=np.float64)
        if matrix.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}")
        if not np.allclose(matrix, matrix.T, atol=1e-9):
            raise ValueError(f"{name} must be symmetric")
        if np.linalg.eigvalsh(matrix)[0] < -1e-9:
            raise ValueError(f"{name} must be positive semidefinite")
        return matrix

    def _model_name(self) -> str:
        """Имя acados-модели (переопределяется в EnvMPCController3D)."""
        return f"belka3d_mpc_{self._body_property_model_suffix()}"

    def _body_property_model_suffix(self) -> str:
        """Return a stable suffix tied only to compiled model structure."""
        if self.robot_params.thruster_model is None:
            raise RuntimeError("RobotParams3D has no thruster model")
        group_masks = np.zeros((len(self.side_force_groups), self.nu), dtype=np.float64)
        for row, group in enumerate(self.side_force_groups):
            group_masks[row, list(group)] = 1.0
        compiled_values = np.concatenate(
            [
                np.array([self.robot_params.m], dtype=np.float64),
                np.asarray(self.robot_params.r_cm_body, dtype=np.float64),
                np.asarray(self.robot_params.I_body, dtype=np.float64).reshape(-1),
                np.asarray(
                    self.robot_params.allocation_matrix,
                    dtype=np.float64,
                ).reshape(-1),
                group_masks.reshape(-1),
                self.active_mask,
                np.array([self.slew_limit], dtype=np.float64),
            ]
        )
        rounded = np.round(compiled_values, decimals=10)
        return hashlib.sha1(rounded.tobytes()).hexdigest()[:10]

    @property
    def _actuator_parameter_size(self) -> int:
        """Runtime vector size for caps, opening shape and two side limits."""
        return self.nu + 4

    def _model_parameter_size(self) -> int:
        """Return actuator runtime parameters for the base MPC."""
        return self._actuator_parameter_size

    def _external_wrench_expr(self, ca, p):
        """CasADi-выражение внешнего wrench [F_world(3), M_world(3)]."""
        del p
        return ca.SX.zeros(6, 1)

    def _runtime_actuator_parameters(self) -> np.ndarray:
        """Return current controller-model actuator coefficients."""
        if self.robot_params.thruster_model is None:
            raise RuntimeError("RobotParams3D has no thruster model")
        opening = self.robot_params.thruster_model.valve_opening_to_fraction
        return np.concatenate(
            [
                self.robot_params.per_thruster_force_caps_n,
                np.array([opening.midpoint, opening.steepness], dtype=np.float64),
                self.robot_params.side_force_limits_n,
            ]
        ).astype(np.float64)

    def _runtime_model_parameters(
        self,
        estimation: Optional[WrenchEstimate3D],
    ) -> np.ndarray:
        """Return actuator parameters; base MPC ignores the wrench estimate."""
        del estimation
        return self._runtime_actuator_parameters()

    def _set_runtime_model_parameters(
        self, estimation: Optional[WrenchEstimate3D]
    ) -> None:
        """Установить runtime-параметры p для всех стадий горизонта."""
        p_vec = self._runtime_model_parameters(estimation)
        if p_vec.shape != (self._model_parameter_size(),):
            raise RuntimeError(
                "runtime model parameter shape mismatch: "
                f"expected {(self._model_parameter_size(),)}, got {p_vec.shape}"
            )
        for stage in range(self.horizon + 1):
            self._acados_solver.set(stage, "p", p_vec)
        limits = self.robot_params.side_force_limits_n.astype(np.float64)
        for stage in range(self.horizon):
            self._acados_solver.constraints_set(stage, "uh", limits)

    def _setup_solver(
        self, state_weight, thruster_weight, thruster_delta_weight, terminal_weight
    ):
        """Построить и скомпилировать acados-солвер: модель, cost, constraints.

        Сначала задаётся дискретная RK4-динамика расширенного состояния [x, u_prev].
        Затем задаются LINEAR_LS cost, box constraints на раскрытия, slew-rate и
        нелинейные ограничения суммы физических сил каждого импеллерного борта.
        """
        import casadi as ca
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

        # ── Символьные переменные ──────────────────────────────────────────
        nx = self.nx  # размерность состояния робота
        nu = self.nu  # размерность управления
        x = ca.SX.sym(
            "x", self.acados_nx
        )  # расширенное состояние [robot_state; u_prev]
        u = ca.SX.sym("u", nu)  # вектор управления: раскрытия клапанов [0, 1]
        np_size = self._model_parameter_size()  # размер runtime-параметров
        p = ca.SX.sym(
            "p", np_size
        )  # actuator params, optionally followed by external wrench
        dt = float(self.dt)  # шаг дискретизации
        m = float(self.robot_params.m)  # масса робота

        # ── Константы модели ───────────────────────────────────────────────
        I_dm = ca.DM(
            np.asarray(self.robot_params.I_body, dtype=float)
        )  # тензор инерции I_body (3×3)
        I_inv_dm = ca.DM(np.linalg.inv(self.robot_params.I_body))  # I_body⁻¹
        c_dm = ca.DM(
            np.asarray(self.robot_params.r_cm_body, dtype=float)
        )  # body-frame вектор от origin к CM
        alloc = ca.DM(
            np.asarray(self.robot_params.allocation_matrix, dtype=float)
        )  # 6×N матрица allocation
        force_caps_n = p[:nu]
        opening_midpoint = p[nu]
        opening_steepness = p[nu + 1]
        side_limits_n = p[nu + 2 : nu + 4]
        desired_thruster_forces = symbolic_desired_thruster_forces(
            ca,
            u,
            force_caps_n,
            opening_midpoint,
            opening_steepness,
            self.active_mask,
        )
        applied_thruster_forces = symbolic_applied_thruster_forces(
            ca,
            desired_thruster_forces,
            self.side_force_groups,
            side_limits_n,
        )

        external_wrench = self._external_wrench_expr(
            ca, p
        )  # [F_ext_world; M_ext_world] — 6-вектор

        # ── Динамика робота (6DoF rigid body + quaternion kinematics) ──────
        def _quat_multiply(q1, q2):
            """Гамильтоново произведение кватернионов q1 ⊗ q2."""
            w1, x1, y1, z1 = (
                q1[0],
                q1[1],
                q1[2],
                q1[3],
            )  # компоненты первого кватерниона
            w2, x2, y2, z2 = (
                q2[0],
                q2[1],
                q2[2],
                q2[3],
            )  # компоненты второго кватерниона
            return ca.vertcat(
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )

        def robot_derivative(robot_state):
            """Непрерывная динамика робота: ẋ = f(x,u).

            Состояние: [p(3), q(4), v(3), ω(3)].
            Уравнения:
              ṗ = v
              q̇ = ½ q ⊗ [0, ω]
              v̇_origin = (R(q) F_body + F_ext) / m − R(q)(ω̇×c + ω×(ω×c))
              ω̇ = I⁻¹ (M_origin − c×F_body + Rᵀ·M_ext_world − ω×Iω)
            """
            q = robot_state[3:7]  # кватернион ориентации [w,x,y,z]
            v = robot_state[7:10]  # линейная скорость world-frame
            omega = robot_state[10:13]  # угловая скорость body-frame ω
            qw, qx, qy, qz = q[0], q[1], q[2], q[3]  # компоненты кватерниона ориентации

            # Кинематика кватерниона
            omega_quat = ca.vertcat(
                0.0, omega[0], omega[1], omega[2]
            )  # pure quaternion угловой скорости
            q_dot = 0.5 * _quat_multiply(q, omega_quat)  # производная кватерниона

            # Wrench от движителей
            wrench_body = alloc @ applied_thruster_forces
            F_body = wrench_body[0:3]  # сила в body-frame
            M_origin_body = wrench_body[3:6]  # момент около body-origin в body-frame

            # Матрица поворота R(q) из body в world
            R = ca.vertcat(  # R_body_to_world rotation matrix
                ca.horzcat(
                    1 - 2 * (qy * qy + qz * qz),
                    2 * (qx * qy - qw * qz),
                    2 * (qx * qz + qw * qy),
                ),
                ca.horzcat(
                    2 * (qx * qy + qw * qz),
                    1 - 2 * (qx * qx + qz * qz),
                    2 * (qy * qz - qw * qx),
                ),
                ca.horzcat(
                    2 * (qx * qz - qw * qy),
                    2 * (qy * qz + qw * qx),
                    1 - 2 * (qx * qx + qy * qy),
                ),
            )
            # Динамика угловой скорости: thrust moment переносится от origin к CM.
            I_omega = ca.mtimes(I_dm, omega)  # I·ω
            omega_cross = ca.cross(omega, I_omega)  # ω × (I·ω)
            M_ext_body = ca.mtimes(
                R.T, external_wrench[3:6]
            )  # внешний момент в body-frame
            M_cm_body = (
                M_origin_body - ca.cross(c_dm, F_body) + M_ext_body
            )  # момент около CM
            net_moment = M_cm_body - omega_cross  # чистый момент
            omega_dot = ca.mtimes(I_inv_dm, net_moment)  # ω̇

            # Динамика body-origin: ускорение CM минус ускорение CM относительно origin.
            a_cm_world = (
                ca.mtimes(R, F_body) + external_wrench[0:3]
            ) / m  # world-frame ускорение CM
            a_offset_body = ca.cross(omega_dot, c_dm) + ca.cross(
                omega, ca.cross(omega, c_dm)
            )  # a_CM-origin
            v_dot = a_cm_world - ca.mtimes(
                R, a_offset_body
            )  # world-frame ускорение origin

            return ca.vertcat(v[0], v[1], v[2], q_dot, v_dot, omega_dot)

        # ── Дискретизация RK4 ──────────────────────────────────────────────
        k1 = robot_derivative(x[:nx])  # первая стадия RK4: f(x)
        k2 = robot_derivative(
            x[:nx] + 0.5 * dt * k1
        )  # вторая стадия RK4: f(x + ½dt·k1)
        k3 = robot_derivative(
            x[:nx] + 0.5 * dt * k2
        )  # третья стадия RK4: f(x + ½dt·k2)
        k4 = robot_derivative(x[:nx] + dt * k3)  # четвертая стадия RK4: f(x + dt·k3)
        next_robot_state_raw = x[:nx] + (dt / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )  # raw x_{k+1}
        q_next = next_robot_state_raw[3:7]  # предсказанный quaternion до нормализации
        q_next_norm = ca.sqrt(ca.sumsqr(q_next) + 1e-12)  # ||q_next||
        next_robot_state = ca.vertcat(
            next_robot_state_raw[0:3],
            q_next / q_next_norm,
            next_robot_state_raw[7:13],
        )  # x_{k+1} с unit quaternion

        # ── Acados-модель ──────────────────────────────────────────────────
        model = AcadosModel()
        model.name = f"{self._model_name()}_h{self.horizon}"
        model.x = x  # вектор состояния
        model.u = u  # вектор управления
        model.p = p  # runtime actuator parameters and optional external wrench
        model.disc_dyn_expr = ca.vertcat(
            next_robot_state, u
        )  # дискретная динамика: xₖ₊₁ = f(xₖ,uₖ), uₖ = uₖ
        side_force_expr = ca.vertcat(
            *[
                ca.sum1(desired_thruster_forces[list(group)])
                for group in self.side_force_groups
            ]
        )
        model.con_h_expr = side_force_expr
        model.con_h_expr_0 = side_force_expr

        # ── OCP описание ───────────────────────────────────────────────────
        ocp = AcadosOcp()  # задача оптимального управления
        ocp.model = model
        ocp.parameter_values = np.zeros(np_size, dtype=np.float64)
        project_root = Path(__file__).resolve().parents[2]
        code_export_directory = project_root / "build" / "acados" / model.name
        json_file = code_export_directory / f"{model.name}.json"
        ocp.code_gen_opts.code_export_directory = str(code_export_directory)
        ocp.code_gen_opts.json_file = str(json_file)
        ocp.solver_options.N_horizon = self.horizon
        ocp.solver_options.tf = self.horizon * self.dt  # горизонт в секундах

        # ── Cost: LINEAR_LS — y = Vx·x + Vu·u, cost = ‖y − yref‖²_W ──────
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        Vx = np.zeros(
            (self.acados_ny, self.acados_nx), dtype=np.float64
        )  # матрица residual-а по x
        Vu = np.zeros((self.acados_ny, nu), dtype=np.float64)  # матрица residual-а по u
        Vx[:nx, :nx] = np.eye(nx, dtype=np.float64)  # y[0:nx] = x
        Vu[nx : nx + nu, :] = np.eye(nu, dtype=np.float64)  # y[nx:nx+nu] = u
        Vx[nx + nu :, nx:] = -np.eye(nu, dtype=np.float64)  # y[nx+nu:] = u − u_prev
        Vu[nx + nu :, :] = np.eye(nu, dtype=np.float64)

        W = np.zeros(
            (self.acados_ny, self.acados_ny), dtype=np.float64
        )  # весовая матрица stage cost
        W[:nx, :nx] = state_weight
        W[nx : nx + nu, nx : nx + nu] = thruster_weight
        W[nx + nu :, nx + nu :] = thruster_delta_weight

        Vx_e = np.zeros((nx, self.acados_nx), dtype=np.float64)  # терминальный residual
        Vx_e[:, :nx] = np.eye(nx, dtype=np.float64)

        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.W = W
        ocp.cost.yref = np.zeros(
            self.acados_ny, dtype=np.float64
        )  # yref обновляется на каждом шаге
        ocp.cost.Vx_e = Vx_e
        ocp.cost.W_e = terminal_weight
        ocp.cost.yref_e = np.zeros(nx, dtype=np.float64)

        # ── Constraints: lbu ≤ u ≤ ubu, lg ≤ C·x + D·u ≤ ug ────────────────
        ocp.constraints.x0 = np.zeros(self.acados_nx, dtype=np.float64)
        ocp.constraints.idxbu = np.arange(nu)  # индексы box-constraints по u
        ocp.constraints.lbu = np.zeros(nu, dtype=np.float64)
        ocp.constraints.ubu = self.active_mask.copy()

        # Общие линейные constraints: только slew-rate раскрытия.
        ng = nu
        C = np.zeros((ng, self.acados_nx), dtype=np.float64)  # матрица при x
        D = np.zeros((ng, nu), dtype=np.float64)  # матрица при u
        lg = np.empty(ng, dtype=np.float64)  # нижняя граница
        ug = np.empty(ng, dtype=np.float64)  # верхняя граница
        D[:nu, :] = np.eye(nu, dtype=np.float64)  # строки slew: u[k] − u[k−1]
        C[:nu, nx:] = -np.eye(nu, dtype=np.float64)  # C·x = −u_prev
        lg[:nu] = -self.slew_limit  # −Δu_max ≤ u − u_prev ≤ Δu_max
        ug[:nu] = self.slew_limit
        ocp.constraints.C = C
        ocp.constraints.D = D
        ocp.constraints.lg = lg
        ocp.constraints.ug = ug
        ocp.constraints.lh = np.zeros(len(self.side_force_groups), dtype=np.float64)
        ocp.constraints.uh = np.ones(
            len(self.side_force_groups),
            dtype=np.float64,
        )
        ocp.constraints.lh_0 = np.zeros(
            len(self.side_force_groups),
            dtype=np.float64,
        )
        ocp.constraints.uh_0 = np.ones(
            len(self.side_force_groups),
            dtype=np.float64,
        )

        # ── Solver options ─────────────────────────────────────────────────
        for option, value in self.effective_solver_options.items():
            if hasattr(ocp.solver_options, option):
                setattr(ocp.solver_options, option, value)
            else:
                raise ValueError(f"Unsupported acados solver option: {option}")

        code_export_directory.mkdir(parents=True, exist_ok=True)
        shared_library_pattern = f"*acados_ocp_solver_{model.name}.*"
        shared_library_exists = any(
            path.suffix in {".so", ".dylib", ".dll"}
            for path in code_export_directory.glob(shared_library_pattern)
        )
        reuse_existing = json_file.exists() and shared_library_exists
        if not reuse_existing and not self.build_solver_if_missing:
            raise RuntimeError(
                f"acados solver is not built completely in {code_export_directory}; "
                "run the scenario prebuild command first"
            )
        self._acados_solver = AcadosOcpSolver(
            ocp,
            verbose=False,
            generate=not reuse_existing,
            build=not reuse_existing,
        )
        self._set_runtime_model_parameters(None)

    def __call__(
        self,
        t: float,
        current_forces: ValveOpenings,
        estimated_state: RobotState3D,
        estimation: WrenchEstimate3D | None,
    ) -> ValveOpenings:
        """Один MPC-шаг: reference horizon, OCP solve and history write.

        Reference horizon строится только из self.planner через общий Controller API.
        """
        first_sample = self.planner.next_target(t, estimated_state)
        reference_states = []
        for stage in range(self.horizon + 1):
            sample = (
                first_sample
                if stage == 0
                else self.planner.target_at_time(t + stage * self.dt)
            )
            reference_states.append(sample.state)

        control = self._solve(
            estimated_state, current_forces, reference_states, estimation
        )
        error_state = estimated_state - reference_states[0]
        self._record_history(
            ts=t,
            state=estimated_state,
            control=control,
            reference_state=reference_states[0],
            real_force=valve_openings_to_forces(
                current_forces,
                self.robot_params,
                active_mask=self.active_mask,
            ),
            error=error_state,
            estimation=estimation,
        )
        return control

    def _solve(
        self,
        state: RobotState3D,
        current_forces: ValveOpenings,
        reference_states: list[RobotState3D],
        estimation: Optional[WrenchEstimate3D] = None,
    ) -> ValveOpenings:
        """Решить OCP с заданным горизонтом reference-состояний.

        1. Фиксирует начальное состояние x0 = [state; u_prev].
        2. Устанавливает yref на каждом этапе горизонта.
        3. Вызывает acados SQP; при неудаче возвращает текущее управление.
        """
        current_control = (
            np.clip(current_forces.to_array(), 0.0, self.max_opening) * self.active_mask
        ).astype(np.float64)
        state_array = self._state_array_for_mpc(
            state
        )  # нормализованное состояние робота
        x0 = np.concatenate((state_array, current_control))  # начальное состояние OCP

        self._acados_solver.constraints_set(0, "lbx", x0)
        self._acados_solver.constraints_set(0, "ubx", x0)
        self._set_runtime_model_parameters(
            estimation
        )  # установить оценку внешнего wrench

        state_refs = self._reference_state_matrix(state_array[3:7], reference_states)

        # Установка reference и warm-start для всех стадий горизонта
        for stage in range(self.horizon):
            yref = np.zeros(self.acados_ny, dtype=np.float64)  # reference для LINEAR_LS
            yref[: self.nx] = state_refs[:, stage]
            self._acados_solver.cost_set(stage, "yref", yref)
            self._acados_solver.set(stage, "x", x0)
            self._acados_solver.set(stage, "u", current_control)

        self._acados_solver.cost_set(self.horizon, "yref", state_refs[:, self.horizon])
        self._acados_solver.set(self.horizon, "x", x0)

        solve_started = time.perf_counter()  # wall-clock start of acados solve
        status = self._acados_solver.solve()
        self.solve_times.append(time.perf_counter() - solve_started)
        self.solve_statuses.append(int(status))
        if status not in (0, 2):  # 0=success, 2=max_iter (решение приемлемо)
            self.last_prediction = None
            self.last_prediction_controls = None
            return ValveOpenings(values=current_control)

        predicted_states = np.column_stack(
            [
                self._acados_solver.get(stage, "x")[: self.nx]
                for stage in range(self.horizon + 1)
            ]
        )
        predicted_controls = np.column_stack(
            [self._acados_solver.get(stage, "u") for stage in range(self.horizon)]
        )
        control_array = (
            np.clip(predicted_controls[:, 0], 0.0, self.max_opening) * self.active_mask
        )
        self.last_prediction = predicted_states
        self.last_prediction_controls = predicted_controls

        return ValveOpenings(values=control_array.astype(float))

    @staticmethod
    def _state_array_for_mpc(state: RobotState3D) -> np.ndarray:
        """Return MPC state vector with a normalized quaternion."""
        x = state.array.astype(np.float64)  # MPC state [p, q, v, omega]
        x[3:7] = quat_normalize(x[3:7])
        return x

    def _reference_state_matrix(
        self, current_q: np.ndarray, reference_states: list[RobotState3D]
    ) -> np.ndarray:
        """Build reference matrix with quaternion signs aligned to the shortest S³ distance."""
        refs = []  # список reference-векторов состояния
        q_anchor = quat_normalize(
            current_q
        )  # предыдущий quaternion для выбора полусферы
        for state in reference_states:
            x_ref = self._state_array_for_mpc(state)  # reference state [p, q, v, omega]
            if float(np.dot(x_ref[3:7], q_anchor)) < 0.0:
                x_ref[3:7] *= -1.0
            q_anchor = x_ref[3:7]
            refs.append(x_ref)
        return np.column_stack(refs)


class EnvMPCController3D(MPCController3D):
    """MPC с оценкой world-frame внешнего wrench как runtime-параметров acados."""

    def _model_name(self) -> str:
        """Вернуть отдельное имя generated acados model для environment-aware 3D MPC."""
        return f"belka3d_env_mpc_{self._body_property_model_suffix()}"

    def _model_parameter_size(self) -> int:
        """Return actuator parameters followed by the external wrench."""
        return self._actuator_parameter_size + 6

    def _external_wrench_expr(self, ca, p):
        """Внешний wrench из runtime-параметра p."""
        start = self._actuator_parameter_size
        return ca.vertcat(*[p[start + index] for index in range(6)])

    def _runtime_model_parameters(
        self,
        estimation: Optional[WrenchEstimate3D],
    ) -> np.ndarray:
        """Append latest world-frame wrench estimate to actuator parameters."""
        wrench = (
            np.zeros(6, dtype=np.float64)
            if estimation is None
            else estimation.to_array().astype(np.float64)
        )
        return np.concatenate([self._runtime_actuator_parameters(), wrench])
