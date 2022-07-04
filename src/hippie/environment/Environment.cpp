/**
 * \file	Environment.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Environment class source file.
 */

#include "Environment.h"
#include "util/Logger.h"
#include <QFile>
#include <QFileInfo>
#include <QStringlist>
#include <QTextStream>

Environment * Environment::instance = nullptr;

bool Environment::filterValue(const QString & value, QString & filteredValue, OptType type) {
    bool valid = true;
    if (type == OPT_INT) {
        int val = value.toInt(&valid);
        filteredValue = QString::number(val);
    }
    else if (type == OPT_FLOAT) {
        float val = value.toFloat(&valid);
        filteredValue = QString::number(val);
    }
    else if (type == OPT_BOOL) {
        if (value == "true" || value == "yes" || value == "on" || value == "1" ||
            value == "TRUE" || value == "YES" || value == "ON") {
            filteredValue = "1";
        }
        else if (value == "false" || value == "no" || value == "off" || value == "0" ||
            value == "FALSE" || value == "NO" || value == "OFF") {
            filteredValue = "0";
        }
        else {
            valid = false;
        }
    }
    else if (type == OPT_VECTOR) {
        QStringList numbers = value.split(QRegExp("(\\ |\\n|\\t)"));
        valid = valid && numbers.size() == 3;
        if (numbers.size() == 3) {
            float val1 = numbers[0].toFloat(&valid);
            float val2 = numbers[1].toFloat(&valid);
            float val3 = numbers[2].toFloat(&valid);
            filteredValue = QString::number(val1) + " " + QString::number(val2) + " " + QString::number(val3);
        }
        else {
            valid = false;
        }
    }
    else {
        filteredValue = value;
        if (value.isEmpty()) valid = false;
    }
    return valid;

}

bool Environment::findOption(const QString & name, Option & option) {
    QVector<Option>::iterator i;
    for (i = options.begin(); i != options.end(); ++i) {
        if (i->name == name) {
            option = *i;
            return true;
        }
    }
    return false;
}

void Environment::registerOption(const QString & name, const QString defaultValue, OptType type) {
    Option opt;
    if (!filterValue(defaultValue, opt.defaultValue, type)) {
        logger(LOG_ERROR) << "ERROR <Environment> Invalid default value for option '" << name << "'.\n";
        exit(EXIT_FAILURE);
    }
    opt.name = name;
    opt.type = type;
    options.push_back(opt);
}

void Environment::registerOption(const QString & name, OptType type) {
    Option opt;
    opt.name = name;
    opt.type = type;
    options.push_back(opt);
}

Environment * Environment::getInstance() {
    if (!instance)
        logger(LOG_WARN) << "WARN <Environment> Environment is not allocated.\n";
    return instance;
}

void Environment::deleteInstance() {
    if (instance) {
        delete instance;
        instance = nullptr;
    }
}

void Environment::setInstance(Environment * instance) {
    Environment::instance = instance;
}

Environment::Environment() {
}

Environment::~Environment() {
}

bool Environment::getIntValue(const QString & name, int & value) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.first().isEmpty()) {
        value = opt.values.first().toInt();
        return true;
    }
    else if (!opt.defaultValue.isEmpty()) {
        value = opt.defaultValue.toInt();
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getFloatValue(const QString & name, float & value) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.first().isEmpty()) {
        value = opt.values.first().toFloat();
        return true;
    }
    else if (!opt.defaultValue.isEmpty()) {
        value = opt.defaultValue.toFloat();
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getBoolValue(const QString & name, bool & value) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.first().isEmpty()) {
        value = (bool)opt.values.first().toInt();
        return true;
    }
    else if (!opt.defaultValue.isEmpty()) {
        value = (bool)opt.defaultValue.toInt();
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getVectorValue(const QString & name, QVector3D & value) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.first().isEmpty()) {
        QStringList nums = opt.values.first().split(QRegExp("(\\ |\\n|\\t)"));
        value = QVector3D(nums[0].toFloat(), nums[1].toFloat(), nums[2].toFloat());
        return true;
    }
    else if (!opt.defaultValue.isEmpty()) {
        QStringList nums = opt.defaultValue.split(QRegExp("(\\ |\\n|\\t)"));
        value = QVector3D(nums[0].toFloat(), nums[1].toFloat(), nums[2].toFloat());
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getStringValue(const QString & name, QString & value) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.first().isEmpty()) {
        value = opt.values.first();
        return true;
    }
    else if (!opt.defaultValue.isEmpty()) {
        value = opt.defaultValue;
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getIntValues(const QString & name, QVector<int> & values) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.first().isEmpty()) {
        values.clear();
        QStringList::iterator i;
        for (i = opt.values.begin(); i != opt.values.end(); ++i)
            values.push_back(i->toInt());
        return true;
    }
    else if (!opt.defaultValue.isEmpty()) {
        values.push_back(opt.defaultValue.toInt());
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getFloatValues(const QString & name, QVector<float> & values) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.first().isEmpty()) {
        values.clear();
        QStringList::iterator i;
        for (i = opt.values.begin(); i != opt.values.end(); ++i)
            values.push_back(i->toFloat());
        return true;
    }
    else if (!opt.defaultValue.isEmpty()) {
        values.push_back(opt.defaultValue.toInt());
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getBoolValues(const QString & name, QVector<bool> & values) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.first().isEmpty()) {
        values.clear();
        QStringList::iterator i;
        for (i = opt.values.begin(); i != opt.values.end(); ++i)
            values.push_back((bool)i->toInt());
        return true;
    }
    else if (!opt.defaultValue.isEmpty()) {
        values.push_back((bool)opt.defaultValue.toInt());
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getVectorValues(const QString & name, QVector<QVector3D> & values) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.first().isEmpty()) {
        values.clear();
        QStringList::iterator i;
        for (i = opt.values.begin(); i != opt.values.end(); ++i) {
            QStringList nums = i->split(QRegExp("(\\ |\\n|\\t)"));
            values.push_back(QVector3D(nums[0].toFloat(), nums[1].toFloat(), nums[2].toFloat()));
        }
        return true;
    }
    else if (!opt.defaultValue.isEmpty()) {
        QStringList nums = opt.defaultValue.split(QRegExp("(\\ |\\n|\\t)"));
        values.push_back(QVector3D(nums[0].toFloat(), nums[1].toFloat(), nums[2].toFloat()));
        return true;
    }
    else {
        return false;
    }
}

bool Environment::getStringValues(const QString & name, QStringList & values) {
    Option opt;
    if (!findOption(name, opt)) return false;
    if (!opt.values.empty() && !opt.values.first().isEmpty()) {
        values = opt.values;
        return true;
    }
    else if (!opt.defaultValue.isEmpty()) {
        values.push_back(opt.defaultValue);
        return true;
    }
    else {
        return false;
    }
}

bool Environment::parse(int argc, char ** argv, bool useExePath) {
    QString envFilename, path;
    for (int i = 1; i < argc; ++i) {
        QString arg = argv[i];
        int index = arg.lastIndexOf(".env");
        if (index != -1) {
            envFilename = argv[i];
            break;
        }
    }
    if (envFilename.isEmpty())
        envFilename = "default.env";
    if (useExePath) {
        QString path = QFileInfo(QFile(argv[0])).absolutePath();
        envFilename = path + envFilename;
    }
    return readEnvFile(envFilename);
}

bool Environment::readEnvFile(const QString & filename) {

    QFile file(filename);

    if (!file.exists()) {
        logger(LOG_WARN) << "WARN <Environment> File '" << filename << "' does not exist.\n";
        return false;
    }

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        logger(LOG_WARN) << "WARN <Environment> Cannot open file '" << filename << "'.\n";
        return false;
    }

    QTextStream in(&file);
    QString blockName;
    QString last;
    QString value;
    bool block = false;
    int lineNumber = 0;

    while (!in.atEnd()) {

        // Read line.
        QString line = in.readLine();
        ++lineNumber;

        // Get rid of comments.
        int commentIndex = line.indexOf("#");
        if (commentIndex != -1) {
            line = line.mid(0, commentIndex);
        }

        // Split line.
        QStringList words = line.split(QRegExp("(\\ |\\n|\\t)"), QString::SkipEmptyParts);

        // Process words.
        for (int i = 0; i < words.size(); ++i) {

            if (words[i] == "{") {
                blockName = last;
                block = true;
            }

            else if (words[i] == "}") {
                if (blockName.isEmpty()) {
                    logger(LOG_WARN) << "WARN <Environment> Unpaired } in '" << filename << "' (line " << line << ").\n";
                    file.close();
                    return false;
                }
                block = false;
                blockName.clear();
            }

            else if (block) {

                bool found = false;
                QString optionName = blockName + "." + words[i];
                QVector<Option>::iterator j;
                for (j = options.begin(); j != options.end(); ++j) {
                    if (optionName == j->name) {
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    logger(LOG_WARN) << "WARN <Environment> Unknown option '" <<
                        optionName << "' in environment file '" << filename << "' (line " << lineNumber << ").\n";
                    file.close();
                    return false;
                }
                else {

                    switch (j->type) {

                    case OPT_INT: {
                        if (i + 1 >= words.size() || !filterValue(words[++i], value, OPT_INT)) {
                            logger(LOG_WARN) << "WARN <Environment> Mismatch in int variable " <<
                                optionName << " in environment file '" << filename << "' (line " << lineNumber << ").\n";
                            file.close();
                            return false;
                        }
                        else {
                            j->values.push_back(value);
                        }
                        break;
                    }

                    case OPT_FLOAT: {
                        if (i + 1 >= words.size() || !filterValue(words[++i], value, OPT_FLOAT)) {
                            logger(LOG_WARN) << "WARN <Environment> Mismatch in float variable " <<
                                optionName << " in environment file '" << filename << "' (line " << lineNumber << ").\n";
                            file.close();
                            return false;
                        }
                        else {
                            j->values.push_back(value);
                        }
                        break;
                    }

                    case OPT_BOOL: {
                        if (i + 1 >= words.size() || !filterValue(words[++i], value, OPT_BOOL)) {
                            logger(LOG_WARN) << "WARN <Environment> Mismatch in bool variable " <<
                                optionName << " in environment file '" << filename << "' (line " << lineNumber << ").\n";
                            file.close();
                            return false;
                        }
                        else {
                            j->values.push_back(value);
                        }
                        break;
                    }

                    case OPT_VECTOR: {
                        if (i + 3 >= words.size() || !filterValue(words[i + 1] + " " + words[i + 2] + " " + words[i + 3], value, OPT_VECTOR)) {
                            logger(LOG_WARN) << "WARN <Environment> Mismatch in vector variable " <<
                                optionName << " in environment file '" << filename << "' (line " << lineNumber << ").\n";
                            file.close();
                            return false;
                        }
                        else {
                            j->values.push_back(value);
                        }
                        i += 3;
                        break;
                    }

                    case OPT_STRING: {
                        if (i + 1 >= words.size() || !filterValue(words[++i], value, OPT_STRING)) {
                            logger(LOG_WARN) << "WARN <Environment> Mismatch in string variable " <<
                                optionName << " in environment file '" << filename << "' (line " << lineNumber << ").\n";
                            file.close();
                            return false;
                        }
                        else {
                            j->values.push_back(value);
                        }
                        break;
                    }

                    }

                }

            }

            last = words[i];

        }

    }

    file.close();
    return true;

}
