class AdmittanceController:
    def __init__(self, Kd, Bd, Md, target_force,
                init_position=-251.0, target_position=-280.0,
                max_vel=5.0, max_acc=20.0):
        self.Kd = Kd
        self.Bd = Bd
        self.Md = Md
        self.target_force = target_force
        self.target_position = target_position

        # 命令轨迹初始
        self.command_pos = init_position
        self.command_vel = 0.0
        # 限幅
        self.max_vel = max_vel
        self.max_acc = max_acc

    def update(self, current_force, dt):
        # 1. 计算误差
        #    弹簧力：当 command_pos > target_position 时，pos_err>0 → 拉向下（负方向）
        pos_err   = self.command_pos - self.target_position
        vel       = self.command_vel
        force_err = self.target_force - current_force # 在未实现按压时，即机械臂向下探索阶段，target_force < current_force, force_err < 0(机械臂按压后力变小)

        # 2. 模型： M a + B v + K pos_err = force_err
        desired_acc1 = (force_err - self.Bd * vel - self.Kd * pos_err) / self.Md
        # 限幅
        desired_acc = max(min(desired_acc1, self.max_acc), -self.max_acc)

        # 3. 半隐式欧拉积分
        vel_next = vel + desired_acc * dt
        vel_next = max(min(vel_next, self.max_vel), -self.max_vel)
        pos_next = self.command_pos + vel_next * dt

        # 4. 只保留下限（目标位置），防止穿透
        if pos_next < self.target_position:
            pos_next = self.target_position

        # 5. 更新状态
        self.command_vel = vel_next
        self.command_pos = pos_next

        # 调试
        print(f"[DEBUG] F_err={force_err:.2f}N, P_err={pos_err:.2f}mm, "
            f"a={desired_acc:.2f}mm/s², a1={desired_acc1:.2f}mm/s², v={vel_next:.2f}mm/s, cmd_z={pos_next:.2f}mm")
        return self.command_pos