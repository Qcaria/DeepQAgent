classdef GridWorld < handle
    %GRIDWORLD Creates a gridworld environment. This environment includes
    %a player, one piece of food and an enemy.

    properties
        size
        player
        enemy
        food

        FOOD_REWARD = 10;
        MOVE_PENALTY = -1;
        ENEMY_PENALTY = -10;
        
        C_PLAYER = [10, 50, 255]/255; 
        C_FOOD = [0, 255, 70]/255;
        C_ENEMY = [255, 0, 100]/255;

        ACTION_SPACE = 4;
        OBSERVATION_SPACE
        OBSERVATION_MIN
        OBSERVATION_MAX

        step_counter
        MAX_STEPS = 200;
    end
    
    methods
        function obj = GridWorld(size)
            % GridWorld constructor. Size corresponds to the grid side
            % length.
            obj.OBSERVATION_SPACE = ones(1, 4)*(2*size-1);
            obj.OBSERVATION_MIN = -ones(1, 4)*(size-1);
            obj.OBSERVATION_MAX = ones(1, 4)*(size-1);

            obj.size = size;
            obj.reset();
        end

        function img = gen_img(obj)
            % Generates an image of the GridWorld
            img = zeros(obj.size, obj.size, 3);
            img(obj.food.y, obj.food.x, :) = obj.C_FOOD;
            img(obj.enemy.y, obj.enemy.x, :) = obj.C_ENEMY;
            img(obj.player.y, obj.player.x, :) = obj.C_PLAYER;
        end

        function render(obj, delay)
            % Plots an image of the GridWorld and delays for a specified
            % amount
            image(obj.gen_img());
            axis square
            axis off

            pause(delay);
        end

        function observation = observe(obj)
            observation = [obj.food - obj.player, obj.enemy - obj.player];
        end

        function [reward, observation, done] = act(obj, choice)
            % Moves the player according to action, and returns a reward,
            % an observation (the relative position between player
            % and enemy/food separated by rows) and if the episode is finished.    
            obj.player.action(choice);
            observation = obj.observe();

            if obj.player == obj.food
                reward = obj.FOOD_REWARD;
                done = true;
            elseif obj.player == obj.enemy               
                reward = obj.ENEMY_PENALTY;
                done = true;
            else
                reward = obj.MOVE_PENALTY;
                done = false;
            end

            obj.step_counter = obj.step_counter + 1;
            if obj.step_counter >= obj.MAX_STEPS
                done = true;
            end
        end

        function observation = reset(obj)
            obj.step_counter = 0;

            obj.player = Blob(obj.size);

            obj.food = Blob(obj.size);
            while obj.food == obj.player
                obj.food = Blob(obj.size);
            end

            obj.enemy = Blob(obj.size);
            while (obj.enemy == obj.player) || (obj.enemy == obj.food)
                obj.enemy = Blob(obj.size);
            end

            observation = obj.observe();
        end
    end
end

