clear;

EPISODES = 3000;
RENDER_EVERY = 15;
epsilon = 0.1;

env = GridWorld(10);

layers = [featureInputLayer(numel(env.OBSERVATION_SPACE), Normalization="none")
          fullyConnectedLayer(20)
          reluLayer
          fullyConnectedLayer(20)
          reluLayer
          fullyConnectedLayer(env.ACTION_SPACE)];
          %regressionLayer];

%net = trainNetwork(array2table(rand([1, numel(env.OBSERVATION_SPACE) + env.ACTION_SPACE])), layers, trainingOptions("adam", "Verbose",false));

net = dlnetwork(layers);
agent = DQNAgent2(net, env.ACTION_SPACE, env.OBSERVATION_SPACE, 10000, 100);

for k = 1:EPISODES
    render = false;
    disp(['EPISODE: ', num2str(k)])
    if mod(k, RENDER_EVERY) == 0
        render = true;
    end
    figure(2)
    reward(k) = agent.train_episode(env, epsilon, render, 0.1);
    figure(4)
    plot(movmean(reward, 10))
    drawnow
end