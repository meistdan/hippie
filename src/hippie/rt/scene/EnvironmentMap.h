#ifndef _ENVIRONMENT_MAP_H_
#define _ENVIRONMENT_MAP_H_

#include "gpu/Texture.h"
#include "util/Math.h"

class EnvironmentMap {

private:

    Vec2i size;
    Texture texture;

public:

    EnvironmentMap(void);
    ~EnvironmentMap(void);

    void set(const QString & filename);
    void clear(void);

    const Vec2i & getSize(void) const;
    hipTextureObject_t getTextureObject(void);

    friend class SceneLoader;

};

#endif /* _ENVIRONMENT_MAP_H_ */
