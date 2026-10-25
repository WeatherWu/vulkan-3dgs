#include "training_sort_replay_reference.hpp"

#include <iostream>

using namespace sort_replay;
int main() {
    try {
        TrainingScanlineCapture source;
        source.header.gaussianCount = 3;
        source.header.width = 32;
        source.header.height = 16;
        source.header.itemCount = 3;
        source.header.tileCount = 2;
        source.projected.resize(3);
        source.projected[0].centerRadius.z = 2;
        source.projected[1].centerRadius.z = 1;
        source.projected[2].centerRadius.z = 1;
        source.items = {1, 0, 0};
        source.ranges = {{0, 2, 0, 0}, {2, 1, 0, 0}};
        Inputs input;
        input.header.source = source.header;
        input.header.capacity = 8;
        input.projected = source.projected;
        input.gaussianRanges = {{0, 2}, {2, 1}, {3, 0}};
        input.items = {0, 0, 1};
        input.depthKeys = {depthKey(2), depthKey(2), depthKey(1)};
        input.tileKeys = {0, 1, 0};
        input.indices = {0, 1, 2};
        verify(input, source);
        check(serialize(input).size() == 104 + 56 * 3 + 16 * 3, "Serialized size mismatch");
        auto reject = [&](auto mutate) {
            auto invalid = input;
            mutate(invalid);
            bool rejected = false;
            try { verify(invalid, source); } catch (const std::runtime_error&) { rejected = true; }
            check(rejected, "Invalid raw sort input accepted");
        };
        reject([](Inputs& x) { x.header.version = 2; });
        reject([](Inputs& x) { x.header.source.frame = 1; });
        reject([](Inputs& x) { x.header.capacity = 2; });
        reject([](Inputs& x) { x.projected[0].centerRadius.z = 5; });
        reject([](Inputs& x) { x.gaussianRanges[2].x = 2; });
        reject([](Inputs& x) { x.items[2] = 2; });
        reject([](Inputs& x) { x.depthKeys[0] = 0; });
        reject([](Inputs& x) { x.indices[0] = 2; });
        reject([](Inputs& x) { x.tileKeys[0] = 2; });
        reject([](Inputs& x) { x.tileKeys[0] = 1; });
        std::cout << "Sort replay CPU contracts passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
