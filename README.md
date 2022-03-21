# policy-gradient-from-scratch
From scratch implementation of Online DDPG (Actor Critic w/ state value estimator)

Both models are implemented in Pytorch. 

## Calculating the Gradients correctly

Obtaining the correct gradient is not trivial in online reinforcement learning.


```python

    #run episode, update online
    for step in tqdm(range(MAX_STEPS)):
        
        #get action and log probability
        action, lp = select_action(policy_network, state)

        #step with action
        new_state, reward, done = env.step(action)

        #update episode score
        score += reward

        #get state value of current state
        state_tensor = torch.from_numpy(state).float().to(DEVICE)
        state_val = stateval_network(state_tensor)

        #get state value of next state
        new_state_tensor = torch.from_numpy(new_state).float().to(DEVICE)  
        new_state_val = stateval_network(new_state_tensor)

        val_loss = F.smooth_l1_loss(reward + DISCOUNT_FACTOR * new_state_val, state_val)
        val_loss *= I

        #calculate policy loss
        advantage = reward + DISCOUNT_FACTOR * new_state_val.item() - state_val.item()
        policy_loss = -lp.mean() * advantage
        policy_loss *= I

        #Backpropagate policy
        policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_([p for g in policy_optimizer.param_groups for p in g["params"]], .5) 
        policy_optimizer.step()

        #Backpropagate value
        stateval_optimizer.zero_grad()
        val_loss.backward(retain_graph=False)
        stateval_optimizer.step()
        
        
 ```
