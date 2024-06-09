#include <bits/stdc++.h>
using namespace std;


enum class Direction { N, S, E, W, Up, Down };

class GalacticSpaceCraft {
public:
    GalacticSpaceCraft(int startX, int startY, int startZ, Direction startDirection)
        : x(startX), y(startY), z(startZ), direction(startDirection) {}

    void move(char command) {
        switch (command) {
            case 'f':
                moveForward();
                break;
            case 'b':
                moveBackward();
                break;
            case 'l':
                turnLeft();
                break;
            case 'r':
                turnRight();
                break;
            case 'u':
                turnUp();
                break;
            case 'd':
                turnDown();
                break;
        }
    }

    void executeCommands(const std::vector<char>& commands) {
        for (char command : commands) {
            move(command);
        }
    }

    void printPositionAndDirection() const {
        cout << "Final Position: (" << x << ", " << y << ", " << z << ")\n";
        cout << "Final Direction: " << directionToString(direction) << "\n";
    }

private:
    int x, y, z;
    Direction direction;

    void moveForward() {
        switch (direction) {
            case Direction::N:
                y++;
                break;
            case Direction::S:
                y--;
                break;
            case Direction::E:
                x++;
                break;
            case Direction::W:
                x--;
                break;
            case Direction::Up:
                z++;
                break;
            case Direction::Down:
                z--;
                break;
        }
    }

    void moveBackward() {
        switch (direction) {
            case Direction::N:
                y--;
                break;
            case Direction::S:
                y++;
                break;
            case Direction::E:
                x--;
                break;
            case Direction::W:
                x++;
                break;
            case Direction::Up:
                z--;
                break;
            case Direction::Down:
                z++;
                break;
        }
    }

    void turnLeft() {
        switch (direction) {
            case Direction::N:
                direction = Direction::W;
                break;
            case Direction::S:
                direction = Direction::E;
                break;
            case Direction::E:
                direction = Direction::N;
                break;
            case Direction::W:
                direction = Direction::S;
                break;
            case Direction::Up:
            case Direction::Down:
                // No change in direction for up/down movements
                break;
        }
    }

    void turnRight() {
        switch (direction) {
            case Direction::N:
                direction = Direction::E;
                break;
            case Direction::S:
                direction = Direction::W;
                break;
            case Direction::E:
                direction = Direction::S;
                break;
            case Direction::W:
                direction = Direction::N;
                break;
            case Direction::Up:
            case Direction::Down:
                // No change in direction for up/down movements
                break;
        }
    }

    void turnUp() {
        if (direction == Direction::N || direction == Direction::S) {
            direction = Direction::Up;
        }
    }

    void turnDown() {
        if (direction == Direction::N || direction == Direction::S) {
            direction = Direction::Down;
        }
    }

    static std::string directionToString(Direction dir) {
        switch (dir) {
            case Direction::N:
                return "N";
            case Direction::S:
                return "S";
            case Direction::E:
                return "E";
            case Direction::W:
                return "W";
            case Direction::Up:
                return "Up";
            case Direction::Down:
                return "Down";
        }
        return "Unknown";
    }
};

int main() {
    int startX = 0, startY = 0, startZ = 0;
    Direction startDirection = Direction::N;

    std::vector<char> commands = {'f', 'r', 'u', 'b', 'l'};

    GalacticSpaceCraft spacecraft(startX, startY, startZ, startDirection);
    spacecraft.executeCommands(commands);
    spacecraft.printPositionAndDirection();

    return 0;
}