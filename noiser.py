import random
import configs
import trainingConfigs

class Noiser:
    def __init__(self, type, noise_min_interval, noise_max_interval, noise_min_intensity, noise_max_intensity, frequency, noise_probability):
        self.applyNoise = False
        self.noise_min_interval = noise_min_interval
        self.noise_max_interval = noise_max_interval
        self.noise_min_intensity = noise_min_intensity
        self.noise_max_intensity = noise_max_intensity
        self.frequency = frequency
        self.noise_probability = noise_probability
        self.type = type

    def update_and_apply_noise(self, current_simulation_step, control, speed):

        if (not self.applyNoise and (current_simulation_step * configs.SIM_TICK) % self.frequency == 0 and random.random() <= self.noise_probability):
            self.applyNoise = True
            self.apply_noise_start_step = current_simulation_step
            self.apply_noise_duration = random.randrange(int(self.noise_min_interval / configs.SIM_TICK), int(self.noise_max_interval / configs.SIM_TICK) + 1)
            self.apply_noise_intensity = random.uniform(self.noise_min_intensity, self.noise_max_intensity)
            self.apply_noise_sign = 1 if random.random() < 0.5 else -1


        if self.applyNoise:

            if self.type == "lateral":
                
                noise = self.apply_noise_sign * self.apply_noise_intensity * (25.0 / (1 * speed + 25.0)) * max(0, 1 - abs(2 * (current_simulation_step - self.apply_noise_start_step)/ self.apply_noise_duration- 1))
                noise = self.apply_noise_sign * self.apply_noise_intensity * max(0.0, 1.0 - min(speed / trainingConfigs.MAX_FORWARD_SPEED, 1.0)) * max(0, 1 - abs(2 * (current_simulation_step - self.apply_noise_start_step)/ self.apply_noise_duration- 1))
                noisy_steer = control.steer + noise
                noisy_steer = max(-1.0, min(noisy_steer, 1.0))
                control.steer = noisy_steer

            elif self.type == "longitudinal":

                noise = self.apply_noise_sign * self.apply_noise_intensity * max(0, 1 - abs(2 * (current_simulation_step - self.apply_noise_start_step)/ self.apply_noise_duration- 1))
                
                if noise > 0:
                    noisy_throttle = control.throttle + noise
                    control.throttle = max(0.0, min(noisy_throttle, 1.0))
                else:
                    noisy_brake = control.brake + abs(noise)
                    control.brake = max(0.0, min(noisy_brake, 1.0))

            if current_simulation_step - self.apply_noise_start_step >= self.apply_noise_duration:
                self.applyNoise = False

        return control

