/**
 * \file	Environment.h
 * \author	Daniel Meister
 * \date	2014/08/19
 * \brief	Environment class header file.
 */

#ifndef _ENVIRONMENT_H_
#define _ENVIRONMENT_H_

#include <QStringList>
#include <QVector>
#include <QVector3D>

enum OptType {
    OPT_INT,
    OPT_FLOAT,
    OPT_BOOL,
    OPT_VECTOR,
    OPT_STRING
};

struct Option {
    OptType type;
    QString name;
    QStringList values;
    QString defaultValue;
};

class Environment {

private:

    static Environment * instance;

    QVector<Option> options;
    int numberOfOptions;

    bool filterValue(const QString & value, QString & filteredValue, OptType type);
    bool findOption(const QString & name, Option & option);

protected:

    void registerOption(const QString & name, const QString defaultValue, OptType type);
    void registerOption(const QString & name, OptType type);
    virtual void registerOptions(void) = 0;

public:

    static Environment * getInstance(void);
    static void deleteInstance(void);
    static void setInstance(Environment * instance);

    Environment(void);
    virtual ~Environment(void);

    bool getIntValue(const QString & name, int & value);
    bool getFloatValue(const QString & name, float & value);
    bool getBoolValue(const QString & name, bool & value);
    bool getVectorValue(const QString & name, QVector3D & value);
    bool getStringValue(const QString & name, QString & value);

    bool getIntValues(const QString & name, QVector<int> & values);
    bool getFloatValues(const QString & name, QVector<float> & values);
    bool getBoolValues(const QString & name, QVector<bool> & values);
    bool getVectorValues(const QString & name, QVector<QVector3D> & values);
    bool getStringValues(const QString & name, QStringList & values);

    bool parse(int argc, char ** argv, bool useExePath);
    bool readEnvFile(const QString & filename);

};

#endif /* _ENVIRONMENT_H_ */
