#ifndef RENDERER_HPP
#define RENDERER_HPP

#include <vector>

struct Renderer;

Renderer* renderer_create(int w, int h);
void renderer_destroy(Renderer* r);

void render(Renderer* r, std::vector<unsigned char> &pixels);

#endif