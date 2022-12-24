close all; clear;

EPISODES = 50000;
RENDER_EVERY = 5000;

%epsilon = 0;

EPSILON0 = 0.2;
START_EPSILON = 1;
FINISH_EPSILON = round(EPISODES/2);

env = GridWorld(5);
agent = QAgent(env.ACTION_SPACE, env.OBSERVATION_MIN, env.OBSERVATION_MAX);
eps_list = [];

for k = 1:EPISODES
    if (k >= START_EPSILON) && (k <= FINISH_EPSILON)
        epsilon = EPSILON0*(1 - (k-START_EPSILON)/(FINISH_EPSILON-START_EPSILON));
    end

    render = false;
    if mod(k, RENDER_EVERY) == 0
        disp(['Episode: ', num2str(k)]);
        render = true;
    end
    [total_reward(k), ~] = agent.train_episode(env, epsilon, render, 1/15);
end

plot(movmean(total_reward, 500))