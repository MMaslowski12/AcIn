from World import World
from agent import Agent

def main():
    world = World()
    agent = Agent(world=world)
    
    # Print initial state
    complexity = agent._get_complexity()
    accuracy = agent._get_accuracy(world.observe())
    print(f"Initial state:")
    print(f"Complexity: {complexity}")
    print(f"Accuracy: {accuracy}")
    print(f"VFE: {agent.get_vfe()}")
    print(f"Surprise: {agent.get_surprise()}")
    print("--------------------------------")
    
    # Run simulation
    for i in range(1000):
        agent.step()

    # Print final state
    print("Final state:")
    print(f"Agent qx.mu: {agent.qx.mu}, Agent qx.sigma: {agent.qx.sigma}")
    print(f"Agent pyx.b0: {agent.pyx.b0}, Agent pyx.b1: {agent.pyx.b1}, Agent pypx_sigma: {agent.pypx_sigma}")
    print(f"Agent px.mu: {agent.px.mu}, Agent px.sigma: {agent.px.sigma}")
    complexity = agent._get_complexity()
    accuracy = agent._get_accuracy(world.observe())
    print(f"Complexity: {complexity}")
    print(f"Accuracy: {accuracy}")
    print(f"VFE: {agent.get_vfe()}")
    print(f"Surprise: {agent.get_surprise()}")

if __name__ == "__main__":
    main() 