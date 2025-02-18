/*
 *  Copyright (C) 2020 Pablo Castells y Javier Sanz-Cruzado
 *
 *  Este código se ha implementado para la realización de las prácticas de
 *  la asignatura "Búsqueda y minería de información" de 4º del Grado en
 *  Ingeniería Informática, impartido en la Escuela Politécnica Superior de
 *  la Universidad Autónoma de Madrid. El fin del mismo, así como su uso,
 *  se ciñe a las actividades docentes de dicha asignatura.
 *
 */
package es.uam.eps.bmi.recsys.data;

/**
 *
 * @author pablo
 * 
 * Para generalizar la lectura de datos de diferentes tipos en ficheros, por ejemplo 
 * características de ítems que pueden utilizar identificadores enteros o string.
 */
public interface Parser<T> {
    public T parse(String value);
}
