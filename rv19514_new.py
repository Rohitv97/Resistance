from player import Bot
from game import State
import random
import operator

class rv19514(Bot):
    def onGameRevealed(self, players, spies):
        """This function will be called to list all the players, and if you're
        a spy, the spies too -- including others and yourself.
        @param players  List of all players in the game including you.
        @param spies    List of players that are spies, or an empty list.
        """

        global suspected
        global sabotage_count

        sabotage_count = 0
        suspected = {}
        for i in players:
            suspected.update({i: 20.0})

        #SPY INFO ------------If i am a spy---------------
        self.spies = spies #List of all spies in the game
        self.other_spies = [s for s in spies if s.index!=self.index] #list of spies other than me
        self.res = [r for r in players if r not in spies] #list of resistance players

        #Res info------------------if i am resistance------------
        self.players = players

    def select(self, players, count):

        ##############BUGGGGGGGGGGGGGGGGGGGGGGGGG##############

        """Pick a sub-group of players to go on the next mission.
        @param players  The list of all players in the game to pick from.
        @param count    The number of players you must now select.
        @return list    The players selected for the upcoming mission.
        """
        global suspected
        #If you are a spy
        if self.spy:
            return random.sample(self.spies, 1) + random.sample(self.players, count-1)
        else:
            return [self] + random.sample(self.players, count-1)

    def onTeamSelected(self, leader, team):
        """Called immediately after the team is selected to go on a mission,
        and before the voting happens.
        @param leader   The leader in charge for this mission.
        @param team     The team that was selected by the current leader.
        """
        self.say("Mission %d team: %s" % (self.game.turn, team))
        self.leader = leader
        self.team = team

    def vote(self, team):
        # Both types of factions have constant behavior on the last try.
        if self.game.tries == 5:
            return not self.spy
        # Spies select any mission with one or more spies on it.
        if self.spy:
            return len([p for p in self.game.team if p in self.spies]) > 0
        # If I'm not on the team, and it's a team of 3...
        if len(self.game.team) == 3 and not self in self.game.team:
            return False
        return True

    def sabotage(self):
        """Decide what to do on the mission once it has been approved.  This
        function is only called if you're a spy, otherwise you have no choice.
        @return bool        Yes to shoot down a mission.
        """
        return True

    def onMissionComplete(self, sabotaged):
        global sabotage_count
        if sabotaged > 0:
            self.say("You're all spies!!")
            sabotage_count = sabotaged
