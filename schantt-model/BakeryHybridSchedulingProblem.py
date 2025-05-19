import numpy as np
import matplotlib.pyplot as plt

class BakeryHybridSchedulingProblem(Problem):
    def __init__(self, user_sequence, recipe_id_to_index, machines_per_stage, processing_times, changeover_times, batch_sizes, tact_times, debug=False):
        self.seq_length = len(user_sequence)
        self.user_sequence = np.array(user_sequence)
        self.recipe_id_to_index = recipe_id_to_index
        self.n_stages = len(machines_per_stage)
        self.machines_per_stage = machines_per_stage
        self.max_makespan = 0
        self.max_machines = max(machines_per_stage)
        self.debug = debug
        n_var = self.seq_length + (self.seq_length * self.n_stages)
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_constr=0,
            xl=[0] * n_var,
            xu=[self.seq_length - 1] * self.seq_length + [m - 1 for m in machines_per_stage for _ in range(self.seq_length)],
            type_var=int
        )
        self.processing_times = processing_times
        self.changeover_times = changeover_times
        self.batch_sizes = batch_sizes
        self.tact_times = tact_times

    def _evaluate(self, X, out, *args, **kwargs):
        makespans = np.array([self.calculate_makespan(x, store_best=False) for x in X])
        out["F"] = makespans

    def calculate_makespan(self, x, store_best=True):
        perm = x[:self.seq_length]
        machine_choices = x[self.seq_length:].reshape(self.seq_length, self.n_stages)
        seq = self.user_sequence[perm]
        start_times = np.zeros((self.seq_length, self.n_stages))
        end_times = np.zeros((self.seq_length, self.n_stages))
        changeover_times_array = np.zeros((self.seq_length, self.n_stages))
        machine_free_times = np.zeros((self.n_stages, self.max_machines))

        if self.debug:
            print(f"Perm: {perm}, Sequence: {[recipe_ids[idx] for idx in seq]}")
            print(f"Machine choices:\n{machine_choices}")

        for i in range(self.seq_length):
            recipe = seq[i]
            prev_recipe = seq[i - 1] if i > 0 else None

            # Compute changeover times
            for s in range(self.n_stages):
                m = machine_choices[i, s]
                if i == 0 or prev_recipe is None:
                    changeover_times_array[i, s] = 0
                else:
                    changeover = self.changeover_times[s, m, prev_recipe, recipe] if prev_recipe != recipe else 0
                    changeover_times_array[i, s] = changeover.item() if isinstance(changeover, np.ndarray) else changeover

            # Compute earliest possible start time for Stage 0
            s = 0
            m = machine_choices[i, s]
            start_times[i, s] = max(0, machine_free_times[s, m])
            if i > 0:
                start_times[i, s] += changeover_times_array[i, s]
            processing_duration = self.processing_times[s, m, recipe]
            batch_delay = (self.batch_sizes[i] - 1) * self.tact_times[recipe]  # Use batch_sizes[i]
            end_times[i, s] = start_times[i, s] + changeover_times_array[i, s] + processing_duration + batch_delay
            machine_free_times[s, m] = end_times[i, s]

            # Compute start and end times for subsequent stages
            for s in range(1, self.n_stages):
                m = machine_choices[i, s]
                prev_stage_end_first = end_times[i, s - 1] - (self.batch_sizes[i] - 1) * self.tact_times[recipe]
                changeover = changeover_times_array[i, s]
                start_times[i, s] = max(prev_stage_end_first + changeover, machine_free_times[s, m])
                if i > 0 and changeover > 0:
                    start_times[i, s] = max(start_times[i, s], machine_free_times[s, m] + changeover)
                processing_duration = self.processing_times[s, m, recipe]
                batch_delay = (self.batch_sizes[i] - 1) * self.tact_times[recipe]
                end_times[i, s] = start_times[i, s] + changeover_times_array[i, s] + processing_duration + batch_delay
                machine_free_times[s, m] = end_times[i, s]

            # Postponement: Adjust Stage 0 start time if necessary, but minimize delays
            if i > 0:
                postponement_candidates = []
                for s in range(1, self.n_stages):
                    m_prev = machine_choices[i - 1, s]
                    machine_ready = end_times[i - 1, s] + self.changeover_times[s, m_prev, prev_recipe, recipe]
                    cumulative_proc_time = sum(self.processing_times[k, machine_choices[i, k], recipe] for k in range(s))
                    required_start = machine_ready - cumulative_proc_time
                    postponement_candidates.append(required_start)

                if postponement_candidates:
                    original_start = start_times[i, 0]
                    new_start = max(original_start, min(postponement_candidates))
                    if new_start > original_start:
                        # Shift Stage 0 and recalculate subsequent stages
                        start_times[i, 0] = new_start
                        s = 0
                        m = machine_choices[i, s]
                        processing_duration = self.processing_times[s, m, recipe]
                        batch_delay = (self.batch_sizes[i] - 1) * self.tact_times[recipe]
                        end_times[i, s] = start_times[i, s] + changeover_times_array[i, s] + processing_duration + batch_delay
                        machine_free_times[s, m] = end_times[i, s]

                        # Recalculate subsequent stages
                        for s in range(1, self.n_stages):
                            m = machine_choices[i, s]
                            prev_stage_end_first = end_times[i, s - 1] - (self.batch_sizes[i] - 1) * self.tact_times[recipe]
                            changeover = changeover_times_array[i, s]
                            start_times[i, s] = max(prev_stage_end_first + changeover, machine_free_times[s, m])
                            if i > 0 and changeover > 0:
                                start_times[i, s] = max(start_times[i, s], machine_free_times[s, m] + changeover)
                            processing_duration = self.processing_times[s, m, recipe]
                            batch_delay = (self.batch_sizes[i] - 1) * self.tact_times[recipe]
                            end_times[i, s] = start_times[i, s] + changeover_times_array[i, s] + processing_duration + batch_delay
                            machine_free_times[s, m] = end_times[i, s]

            if self.debug:
                for s in range(self.n_stages):
                    m = machine_choices[i, s]
                    print(f"Recipe {i} (ID {recipe_ids[recipe]}): Stage {s}, Machine {m}: "
                          f"Start = {start_times[i, s]}, Changeover = {changeover_times_array[i, s]}, "
                          f"Processing = {self.processing_times[s, m, recipe]}, Batch Delay = {batch_delay}, "
                          f"End = {end_times[i, s]}, Machine free at = {machine_free_times[s, m]}")

        makespan = np.max(end_times)
        if self.debug:
            print(f"Calculated makespan in calculate_makespan: {makespan}")

        if store_best:
            self.best_start_times = start_times
            self.best_end_times = end_times
            self.best_machine_choices = machine_choices
            self.best_sequence_ids = [recipe_ids[idx] for idx in seq]
            self.best_changeover_times = changeover_times_array
            self.max_makespan = makespan

        return makespan

    def plot_gantt_chart(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        stage_labels = [f"Stage {s} (Machine {m})" for s in range(self.n_stages) for m in range(self.machines_per_stage[s])]
        y_positions = np.arange(len(stage_labels))
        colors = {recipe_id: plt.cm.Set3(i) for i, recipe_id in enumerate(np.unique(self.best_sequence_ids))}

        for i in range(self.seq_length):
            recipe_id = self.best_sequence_ids[i]
            for s in range(self.n_stages):
                m = self.best_machine_choices[i, s]
                stage_idx = sum(self.machines_per_stage[:s]) + m
                print(f"Processing Recipe {i} (ID {recipe_id}), Stage {s}, Machine {m}, Stage_idx {stage_idx}")
                if self.best_start_times[i, s] < self.best_end_times[i, s]:
                    print(f"  Plotting: Start={self.best_start_times[i, s]}, End={self.best_end_times[i, s]}, Changeover={self.best_changeover_times[i, s]}")
                    if self.best_changeover_times[i, s] > 0:
                        ax.barh(y_positions[stage_idx], self.best_changeover_times[i, s],
                                left=self.best_start_times[i, s], height=0.5, color='red', edgecolor='black', alpha=0.7,
                                label='Changeover' if i == 0 and s == 0 else "")
                        process_start = self.best_start_times[i, s] + self.best_changeover_times[i, s]
                        print(f"  Changeover bar: Left={self.best_start_times[i, s]}, Width={self.best_changeover_times[i, s]}")
                    else:
                        process_start = self.best_start_times[i, s]
                    process_duration = ((self.best_end_times[i, s] - self.best_start_times[i, s]) -
                                        self.best_changeover_times[i, s])
                    print(f"  Processing bar: Left={process_start}, Width={process_duration}")
                    ax.barh(y_positions[stage_idx], process_duration,
                            left=process_start, height=0.5, color=colors[recipe_id], edgecolor='black')
                    ax.text(process_start + process_duration / 2, y_positions[stage_idx], f"{recipe_id}",
                            ha='center', va='center', color='white', fontweight='bold')
                if self.best_start_times[i, s] >= self.best_end_times[i, s]:
                    print(f"Skipped task at Stage {s}, Machine {m}: Start={self.best_start_times[i, s]}, End={self.best_end_times[i, s]}")

        ax.set_yticks(y_positions)
        ax.set_yticklabels(stage_labels)
        ax.set_xlabel("Time")
        ax.set_title("Batch Production Scheduling - Multi-stage Flowshop Optimization")
        ax.set_xlim(0, self.max_makespan + 10)
        ax.set_ylim(-0.5, len(stage_labels) - 0.5)
        handles = [plt.Rectangle((0,0),1,1, color=colors[rid], label=str(rid)) for rid in colors.keys()]
        handles.append(plt.Rectangle((0,0),1,1, color='red', alpha=0.7, label='Changeover'))
        ax.legend(handles=handles, loc='best', title="Recipe IDs")
        plt.tight_layout()
        plt.show()