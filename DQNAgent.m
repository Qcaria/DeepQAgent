classdef DQNAgent < handle
    %DQNAgent A DQN agent.
    %   Custom DQN agent with memory replay.
    
    properties
        net
        memo
       
        OBSERVATION_SPACE
        ACTION_SPACE
        MAX_MEMO
        MIN_MEMO


        BATCH_SIZE = 64
        DECAY = 0.98
    end
    
    methods
        function obj = DQNAgent(net, ACTION_SPACE, OBSERVATION_SPACE, MAX_MEMO, MIN_MEMO)
            %DQNAgent Construct an instance of this class
            %   Initializes the agent with a critic neural network.
            obj.net = net;
            obj.ACTION_SPACE = ACTION_SPACE;
            obj.OBSERVATION_SPACE = OBSERVATION_SPACE;
            obj.MAX_MEMO = MAX_MEMO;
            obj.MIN_MEMO = MIN_MEMO;

            obj.memo = [];
        end
        
        function remember(obj, past_obs, action, reward, obs)
            n_rows = size(obj.memo, 1);
            
            if n_rows >= obj.MAX_MEMO
                obj.memo(1, :) = [];
            end
            obj.memo = [obj.memo; table(past_obs, action, reward, obs)];
        end

        function [total_reward] = train_episode(obj, env, epsilon, render, delay)
            past_obs = env.reset();
            done = false;
            total_reward = 0;

            while ~done
                if render
                    env.render(delay)
                end
                [~, max_choice] = max(predict(obj.net, past_obs));

                if rand()>epsilon
                    action = max_choice;
                else
                    action = randi(obj.ACTION_SPACE);
                end

                [reward, obs, done] = env.act(action);
                obj.remember(past_obs, action, reward, obs);
                total_reward = total_reward + reward;

                obj.train_net(env)

                past_obs = obs;
            end
        end

        function train_net(obj, env)

            n_rows = size(obj.memo, 1);

            if n_rows > obj.MIN_MEMO
                sample = obj.memo(randperm(n_rows, obj.BATCH_SIZE), :);
    
                past_obs = sample.past_obs;
                past_Qs = predict(obj.net, array2table(past_obs));
                target_Qs = past_Qs;
    
                obs = array2table(sample.obs);
                max_Qs = max(predict(obj.net, obs), [], 2);

                target_Qs(sub2ind(size(sample), 1:obj.BATCH_SIZE, sample.action'))...
                    = sample.reward + max_Qs * obj.DECAY .* (sample.reward == env.MOVE_PENALTY);
    
                batch = array2table([past_obs, target_Qs]);
    
                obj.net = trainNetwork(batch, obj.net.Layers, trainingOptions("adam", "MiniBatchSize", obj.BATCH_SIZE ,"Verbose",false));
            end
        end
    end
end

