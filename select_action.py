def select_action(self, state, eval_mode=False):
    """
    Select action using current policy with improved error handling and debugging
    
    Args:
        state: Current environment state
        eval_mode: If True, use deterministic actions (mean)
    """
    try:
        with torch.no_grad():
            # Verify state dimensions
            if len(state) != self.state_dim:
                print(f"\nWarning: State dimension mismatch in select_action!")
                print(f"Expected: {self.state_dim}, Got: {len(state)}")
                print(f"State content: {state}")
                
                # Try to handle mismatched dimensions
                if len(state) < self.state_dim:
                    # Pad with zeros if state is too small
                    state = np.pad(state, (0, self.state_dim - len(state)), 
                                 'constant', constant_values=0)
                else:
                    # Truncate if state is too large
                    state = state[:self.state_dim]
                print(f"Adjusted state length: {len(state)}")
            
            # Convert to tensor and prepare for network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action distribution parameters
            mean, std = self.actor(state_tensor)
            
            if eval_mode:
                # Use mean action for evaluation
                action = mean
            else:
                # Sample from distribution for training
                dist = Normal(mean, std)
                action = dist.sample()
                
            # Get log probability
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Get value estimate
            value = self.critic(state_tensor)
            
            # Clip actions to valid range
            action = torch.clamp(action, -1.0, 1.0)
            
            # Convert to numpy and remove extra dimensions
            action_np = action.cpu().numpy().squeeze()
            log_prob_np = log_prob.cpu().numpy()
            value_np = value.cpu().numpy().squeeze()
            
            return action_np, log_prob_np, value_np
            
    except Exception as e:
        print(f"\nError in select_action:")
        print(f"State shape: {state.shape if hasattr(state, 'shape') else len(state)}")
        print(f"State type: {type(state)}")
        print(f"Error details: {str(e)}")
        raise