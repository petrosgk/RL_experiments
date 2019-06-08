import sys
import six
import numpy as np

from environments.malmo.missions.base import Mission, MissionEnvironment, MissionStateBuilder


class Rooms(Mission):
  def __init__(self, ms_per_tick):
    mission_name = 'rooms'
    agent_names = ['Agent_1']

    mission_xml = '''<?xml version="1.0" encoding="UTF-8" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" 
xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
              <About>
                <Summary>Find the goal!</Summary>
              </About>
              <ModSettings>
                <MsPerTick>''' + str(ms_per_tick) + '''</MsPerTick>
              </ModSettings>
              <ServerSection>
                  <ServerInitialConditions>
                    <Time>
                      <StartTime>6000</StartTime>
                      <AllowPassageOfTime>false</AllowPassageOfTime>
                    </Time>
                    <Weather>clear</Weather>
                    <AllowSpawning>false</AllowSpawning>
                   </ServerInitialConditions>
                <ServerHandlers>
                  <FileWorldGenerator src="/home/petrosgk/MalmoPlatform/Minecraft/run/saves/Rooms" />
                  <ServerQuitFromTimeUp timeLimitMs="1000000" description="out_of_time"/>
                  <ServerQuitWhenAnyAgentFinishes />
                </ServerHandlers>
              </ServerSection>
              <AgentSection mode="Survival">
                <Name>''' + agent_names[0] + '''</Name>
                <AgentStart>
                  <Placement x="20.5" y="4.0" z="85.5" yaw="0" pitch="15"/>
                </AgentStart>
                <AgentHandlers>
                  <VideoProducer>
                    <Width>256</Width>
                    <Height>256</Height>
                  </VideoProducer>
                  <ObservationFromFullStats/>
                  <ContinuousMovementCommands turnSpeedDegs="45">
                    <ModifierList type="deny-list">
                      <command>attack</command>
                    </ModifierList>
                  </ContinuousMovementCommands>
                  <RewardForMissionEnd>
                    <Reward description="found_goal" reward="1" />
                    <Reward description="death" reward="-1" />
                  </RewardForMissionEnd>
                  <RewardForSendingCommand reward="-1"/>
                  <AgentQuitFromTouchingBlockType>
                    <Block type="gold_block" description="found_goal" />
                  </AgentQuitFromTouchingBlockType>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

    super(Rooms, self).__init__(mission_name=mission_name, agent_names=agent_names,
                                mission_xml=mission_xml)


# Define the mission environment
class RoomsEnvironment(MissionEnvironment):
  def __init__(self, action_space, mission_name, mission_xml, remotes, state_builder, role=0,
               recording_path=None):
    if action_space == 'discrete':
      actions = ['move 1', 'move 0', 'move -1', 'turn 1.0', 'turn 0', 'turn -1.0']
    elif action_space == 'continuous':
      actions = ['move', 'turn']
    else:
      print('Unknown action space')
      sys.exit()

    super(RoomsEnvironment, self).__init__(mission_name, mission_xml, actions, remotes,
                                           state_builder,
                                           role=role, recording_path=recording_path)

  # Do an action. Supports either a single discrete action or a list/tuple of continuous actions
  def step(self, action):
    if isinstance(action, list) or isinstance(action, tuple):
      assert 0 <= len(action) <= self.available_actions, \
        "action list is not valid (should be of length [0, %d[)" % (
          self.available_actions)
    else:
      action_id = action
      assert 0 <= action_id <= self.available_actions, \
        "action %d is not valid (should be in [0, %d[)" % (action_id,
                                                           self.available_actions)

    if isinstance(action, list) or isinstance(action, tuple):
      # For continuous action space, do the environment action(s) by the amount sent by the agent
      #  for each
      action = [self._actions[i] + ' ' + str(action[i]) for i in range(len(action))]
    else:
      # For discrete action space, do the environment action corresponding to the action id sent
      # by the agent
      action = self._actions[action_id]
      assert isinstance(action, six.string_types)

    if isinstance(action, list):
      for i in range(len(action)):
        self._agent.sendCommand(action[i])
    else:
      if self._previous_action == 'use 1':
        self._agent.sendCommand('use 0')
      self._agent.sendCommand(action)
    self._previous_action = action
    self._action_count += 1

    self._await_next_obs()
    return self.state, sum([reward.getValue() for reward in self._world.rewards]), self.done, {}

  @property
  def abs_max_reward(self):
    return self._abs_max_reward


# Define a mission state builder
class RoomsStateBuilder(MissionStateBuilder):
  """
  Generate RGB frame state resizing to the specified width/height and depth
  """

  def __init__(self, width, height, grayscale):
    assert width > 0, 'width should be > 0'
    assert height > 0, 'height should be > 0'

    self._width = width
    self._height = height
    self._gray = bool(grayscale)

    super(RoomsStateBuilder, self).__init__()

  def build(self, environment):

    img = environment.frame
    obs = environment.world_observations

    if img is not None:
      img = img.resize((self._width, self._height))

      if self._gray:
        img = img.convert('L')
      img = np.array(img)
    else:
      img = np.zeros((self._width, self._height, 1 if self._gray else 3)).squeeze()

    return img, obs