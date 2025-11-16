"""
Human control and demonstration management for F1Tenth RL training.
Handles keyboard and gamepad input, demonstration collection, and pretraining triggers.

GAMEPAD SETUP:
1. Install inputs package: pip install inputs
2. Connect Xbox/PlayStation controller
3. Enable in train.py: ControlHandler(..., use_gamepad=True)

GAMEPAD CONTROLS:
- Left stick X-axis: Steering (analog)
- Right trigger (RT): Throttle (0-100%)
- Left trigger (LT): Brake (not implemented, use B button)
- A button: Toggle demo collection
- B button: Emergency brake
- Y button: Trigger pretraining
- X button: Clear demo buffer

KEYBOARD CONTROLS:
- H: Toggle human control on/off
- LEFT/RIGHT arrows: Steer (binary)
- UP/DOWN arrows: Adjust speed
- SPACE: Emergency brake
- C: Toggle demo collection
- R: Clear demo buffer
- P: Trigger pretraining
"""

import numpy as np
import pyglet.window.key
import threading
from utils import load_buffer_from_pkl, buffer_to_pkl

try:
    from inputs import get_gamepad
    GAMEPAD_AVAILABLE = True
except ImportError:
    GAMEPAD_AVAILABLE = False
    print("Warning: 'inputs' package not found. Gamepad support disabled.")
    print("Install with: pip install inputs")


class ControlHandler:
    """Manages human control, demonstration collection, and keyboard/gamepad input."""
    
    def __init__(self, env, current_map, params, max_demo_buffer=10000, use_gamepad=True):
        """
        Initialize the control handler.
        
        Args:
            env: The F1Tenth gym environment
            current_map: Name of the current map
            max_demo_buffer: Maximum number of demonstrations to store
            use_gamepad: Whether to enable gamepad support (if available)
        """
        self.env = env
        self.current_map = current_map
        self.max_demo_buffer = max_demo_buffer
        self.params = params
        
        # Control state
        self.human_control_enabled = False
        self.controlled_agent_idx = 0
        self.human_steering = 0.0
        self.human_speed = 0.0
        self.forward_speed = 0.0
        self.brake_modifier = 0.0
        self.speed_override = 7.0
        
        # Gamepad state
        self.use_gamepad = use_gamepad and GAMEPAD_AVAILABLE
        self.gamepad_thread = None
        self.gamepad_running = False
        self.stick_deadzone = 0.1  # Deadzone for left stick (%)
        
        # Demonstration state
        self.demonstration_mode = "none"  # "none", "collect", or "pretrain"
        self.demonstration_buffer = self._load_demonstrations(current_map)
        
        # Register keyboard handlers
        self._register_handlers()
        
        # Start gamepad thread if enabled
        if self.use_gamepad:
            self._start_gamepad_thread()
    
    def _load_demonstrations(self, map_name):
        """Load existing demonstrations for the given map."""
        try:
            return load_buffer_from_pkl(f"demonstrations_up_to_gen_{map_name}.pkl")
        except FileNotFoundError:
            print(f"No existing demonstrations found for {map_name}")
            return []
    
    def _register_handlers(self):
        """Register keyboard handlers with the environment renderer."""
        if hasattr(self.env, 'renderer') and self.env.renderer is not None:
            self.env.renderer.on_key_press = self._on_key_press
            self.env.renderer.on_key_release = self._on_key_release
            keyboard_msg = "Keyboard handlers registered."
            gamepad_msg = " Gamepad detected!" if self.use_gamepad else ""
            print(f"Human control: {keyboard_msg}{gamepad_msg} Press 'H' to toggle control.\n")
    
    def _start_gamepad_thread(self):
        """Start background thread to read gamepad events."""
        self.gamepad_running = True
        self.gamepad_thread = threading.Thread(target=self._gamepad_loop, daemon=True)
        self.gamepad_thread.start()
    
    def _gamepad_loop(self):
        """Background thread that continuously reads gamepad events."""
        try:
            while self.gamepad_running:
                try:
                    events = get_gamepad()
                    for event in events:
                        self._handle_gamepad_event(event)
                except Exception as e:
                    if self.gamepad_running:  # Only print if not shutting down
                        print(f"Gamepad error: {e}")
                    break
        except Exception:
            pass  # Gamepad disconnected or not available
    
    def _handle_gamepad_event(self, event):
        """
        Handle a single gamepad event.
        
        Xbox/PlayStation controller mapping:
        - Left stick X-axis (ABS_X): Steering
        - Right trigger (ABS_RZ or ABS_Z): Throttle
        - Left trigger (ABS_Z or ABS_RZ): Brake
        - A button (BTN_SOUTH): Toggle demo collection
        - B button (BTN_EAST): Speed limiter
        - Y button (BTN_NORTH): Trigger pretraining
        - X button (BTN_WEST): Clear demo buffer
        """
        if not self.human_control_enabled:
            return
        
        
        v_max = self.speed_override if self.speed_override is not None else self.params['v_max']
        v_min = self.params['v_min']
        
        # Left stick X-axis for steering (range: -32768 to 32767)
        if event.code == 'ABS_X':
            # Normalize to -1.0 to 1.0
            normalized = -event.state / 32768.0
            # Apply deadzone to filter stick drift
            if abs(normalized) < self.stick_deadzone:
                normalized = 0.0
            
            # Scale to steering range (-0.4 to 0.4)
            self.human_steering = normalized * self.params['s_max']
        
        # Right trigger for throttle (range: 0 to 1023)
        elif event.code in ['ABS_RZ']:
            self.forward_speed = event.state / 1023.0
            self.human_speed = np.clip(v_max * (self.forward_speed - self.brake_modifier), v_min, v_max)
        
        # Left trigger for braking (range: 0 to 1023)
        elif event.code in ['ABS_Z']:
            self.brake_modifier = event.state / 1023.0
            self.human_speed = np.clip(v_max * (self.forward_speed - self.brake_modifier), v_min, v_max)
        
        # Button presses
        elif event.code == 'BTN_SOUTH' and event.state == 1:  # A button
            self._toggle_demo_collection()
        
        elif event.code == 'BTN_EAST' and event.state == 1:  # B button
            self.speed_override = 10.0 if self.speed_override != 10.0 else 7.0
            v_max = self.speed_override if self.speed_override is not None else self.params['v_max']
            self.human_speed = np.clip(v_max * (self.forward_speed - self.brake_modifier), v_min, v_max)
            
            if self.speed_override == 7.0:
                print(f"\n## SPEED LIMITER ENABLED: Max speed set to {self.speed_override}")
            else:
                print(f"\n## SPEED LIMITER DISABLED: Max speed set to {self.speed_override}")
        
        elif event.code == 'BTN_NORTH' and event.state == 1:  # Y button
            self._trigger_pretrain()
        
        elif event.code == 'BTN_WEST' and event.state == 1:  # X button
            self._clear_demo_buffer()
                
    def _toggle_demo_collection(self):
        """Toggle demonstration collection mode."""
        if self.demonstration_mode == "none":
            self.demonstration_mode = "collect"
            print("\n## DEMO COLLECTION MODE: Storing your driving for later training")
        elif self.demonstration_mode == "collect":
            self.demonstration_mode = "none"
            print(f"\n## Demo collection stopped. Collected {len(self.demonstration_buffer)} demonstrations")
    
    def _trigger_pretrain(self):
        """Trigger pretraining from demonstrations."""
        if len(self.demonstration_buffer) >= 100:
            self.demonstration_mode = "pretrain"
            print(f"\n ##  PRETRAIN MODE ENABLED: Will use {len(self.demonstration_buffer)} demos")
        else:
            print(f"\n##  Need at least 100 demos to pretrain (current: {len(self.demonstration_buffer)})")
    
    def _clear_demo_buffer(self):
        """Clear the demonstration buffer."""
        self.demonstration_buffer.clear()
        print("\n##  Demonstration buffer cleared.")
    
    def _on_key_press(self, symbol, modifiers):
        """Handle keyboard press events."""
        # Toggle human control with 'H' key
        if symbol == pyglet.window.key.H:
            self.human_control_enabled = not self.human_control_enabled
            print(f"\n{'='*50}")
            print(f"HUMAN CONTROL: {'ENABLED' if self.human_control_enabled else 'DISABLED'}")
            print(f"{'='*50}\n")
            if self.human_control_enabled:
                print("Controls:")
                print("  LEFT/RIGHT arrows: Steer")
                print("  UP/DOWN arrows: Speed")
                print("  H: Toggle human control off")
                print("  SPACE: Emergency brake")
                print("  C: Toggle demo collection mode")
                print("  R: Clear demonstration buffer")
                print("  P: Pretrain from demonstrations")
                if self.use_gamepad:
                    print("\nGamepad controls:")
                    print("  Left stick: Steering")
                    print("  Right trigger: Throttle")
                    print("  A button: Toggle demo collection")
                    print("  B button: Emergency brake")
                    print("  Y button: Pretrain")
                    print("  X button: Clear buffer")
                print(f"\nCurrent mode: {self.demonstration_mode}\n")
        
        # Toggle demonstration mode with 'C' key
        elif symbol == pyglet.window.key.C:
            self._toggle_demo_collection()
            print()
        
        # Clear demonstration buffer with 'R' key
        elif symbol == pyglet.window.key.R:
            self._clear_demo_buffer()
            print()
        
        # Trigger pretraining with 'P' key
        elif symbol == pyglet.window.key.P:
            self._trigger_pretrain()
            print()
        
        # Only process steering/speed if human control is enabled
        if not self.human_control_enabled:
            return
        
        # Steering controls
        if symbol == pyglet.window.key.LEFT:
            self.human_steering = 3.2
        elif symbol == pyglet.window.key.RIGHT:
            self.human_steering = -3.2
        
        # Speed controls
        if symbol == pyglet.window.key.UP:
            self.human_speed = min(20.0, self.human_speed + 1.0)
        elif symbol == pyglet.window.key.DOWN:
            self.human_speed = max(-5.0, self.human_speed - 1.0)
        elif symbol == pyglet.window.key.SPACE:
            self.human_speed = 0.0
    
    def _on_key_release(self, symbol, modifiers):
        """Handle keyboard release events."""
        if symbol in [pyglet.window.key.LEFT, pyglet.window.key.RIGHT]:
            self.human_steering = 0.0
    
    def override_action(self, scaled_values, action_tensor, params_dict):
        """
        Override agent action with human input if human control is enabled.
        
        Args:
            scaled_values: Tensor of scaled actions (steering velocity, speed)
            action_tensor: Tensor of normalized actions (-1 to 1)
            params_dict: Dictionary of environment parameters
            
        Returns:
            Tuple of (scaled_values, action_tensor) with human overrides applied
        """
        if not self.human_control_enabled:
            return scaled_values, action_tensor
        
        # Override the controlled agent's action with human input
        scaled_values[self.controlled_agent_idx, 0] = self.human_steering
        scaled_values[self.controlled_agent_idx, 1] = self.human_speed
        
        # Recompute action tensor in normalized space for proper storage
        # Note: Environment uses steering angle (s_min/s_max)
        steer_scale = (params_dict['s_max'] - params_dict['s_min']) / 2
        steer_shift = (params_dict['s_max'] + params_dict['s_min']) / 2
        speed_scale = (params_dict['v_max'] - params_dict['v_min']) / 2
        speed_shift = (params_dict['v_max'] + params_dict['v_min']) / 2
        
        action_tensor[self.controlled_agent_idx, 0] = (self.human_steering - steer_shift) / steer_scale
        action_tensor[self.controlled_agent_idx, 1] = (self.human_speed - speed_shift) / speed_scale
        
        return scaled_values, action_tensor
    
    def store_demonstration(self, scan_tensor, state_tensor, action_tensor, just_crashed):
        """
        Store a demonstration if collection mode is active and agent didn't crash.
        
        Args:
            scan_tensor: LIDAR scan tensor
            state_tensor: State tensor (velocities)
            action_tensor: Normalized action tensor
            just_crashed: Boolean array indicating which agents crashed
        """
        if not self.human_control_enabled or self.demonstration_mode != "collect":
            return
        
        if just_crashed[self.controlled_agent_idx]:
            return
        
        # Store observation and human action (normalized)
        demo_obs = {
            'scan': scan_tensor[self.controlled_agent_idx].cpu().numpy(),
            'state': state_tensor[self.controlled_agent_idx].cpu().numpy(),
            'action': action_tensor[self.controlled_agent_idx].cpu().numpy()
        }
        self.demonstration_buffer.append(demo_obs)
        
        if len(self.demonstration_buffer) > self.max_demo_buffer:
            self.demonstration_buffer.pop(0)  # Remove oldest demo
    
    def should_pretrain(self):
        """Check if pretraining should be triggered this generation."""
        return self.demonstration_mode == "pretrain" and len(self.demonstration_buffer) >= 100
    
    def save_demonstrations(self):
        """Save demonstrations to disk."""
        if len(self.demonstration_buffer) > 0:
            filename = f"demonstrations_up_to_gen_{self.current_map}.pkl"
            buffer_to_pkl(self.demonstration_buffer, filename)
            print(f"💾 Saved {len(self.demonstration_buffer)} demonstrations to {filename}")
    
    def reset_pretrain_mode(self):
        """Reset demonstration mode to normal after pretraining."""
        self.demonstration_mode = "none"
    
    def update_map(self, new_map):
        """Update the current map name."""
        self.current_map = new_map
    
    def cleanup(self):
        """Cleanup resources (stop gamepad thread)."""
        if self.use_gamepad and self.gamepad_running:
            self.gamepad_running = False
            if self.gamepad_thread is not None:
                self.gamepad_thread.join(timeout=1.0)
